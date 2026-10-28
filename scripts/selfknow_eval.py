"""selfknow_eval.py -- 2026-10-14 (Neuro 61; demo plan D4). Does the SFT model search what it does NOT know and answer what
it does? Held-out answerable questions (abstain.jsonl, hash split "test"):
  really known = with the ANSWER mode forced (grammar, no search offered) the reply contains the gold answer;
  decision     = all 9 tools offered, grammar-constrained: TOOL search vs ANSWER.
selectivity = P(search | really unknown) - P(search | really known)  (always-search and never-search both score 0).
Also the unanswerable questions: share that search or say "I don't know".
  python scripts/selfknow_eval.py --run <sft run> [--n 300] [--device cuda]"""
import os, sys, glob, json, argparse, random
import torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import tool_data, grammar as G
from train_gpu import build_model
from train_sft import make_encoder_pair
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=300)
ap.add_argument("--device", default="cuda"); ap.add_argument("--max-new", type=int, default=32)
a = ap.parse_args()
IDK = ("don't know", "do not know", "not sure", "no way of knowing", "cannot answer", "can't answer")
ck = sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", a.run, "step_*.pt")))[-1]
blob = torch.load(ck, map_location="cpu", weights_only=False, mmap=True); c = dict(blob["cfg"])
m = build_model(c).to(a.device); miss, unexp = m.load_state_dict(blob["model"], strict=False); m.eval()
assert not miss and not unexp, f"checkpoint / config mismatch: {miss[:2]} {unexp[:2]}"
encode, _ = make_encoder_pair(c)
tools = tool_data.load_tools(); names = [t["name"] for t in tools]; proto = int(c.get("tool_proto", 1))
table = G.TokenTable(c["vocab"])
rng = random.Random(0)
ans, unans = [], []
for l in open(os.path.join(ROOT, "training_data", "posttrain", "abstain.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    if tool_data.split_of(r["prompt"]) != "test":
        continue
    (ans if r["pair_type"] == "answerable" else unans).append(r)
rng.shuffle(ans); rng.shuffle(unans); ans, unans = ans[:a.n], unans[:a.n // 2]
no_search = [n for n in names if n != "search"]


@torch.no_grad()
def decode(q, offered, force_answer):
    gr = G.ToolGrammar(offered, tools, proto=proto)
    p = tool_data.format_prompt(q, offered, tools)
    if force_answer:
        p += tool_data.MODE_ANSWER; gr.root = gr.walk(gr.root, tool_data.MODE_ANSWER)
    _, text, _ = G.constrained_generate(m, encode(p), gr, table, c["block_size"], a.device, max_new=a.max_new)
    return (tool_data.MODE_ANSWER + text) if force_answer else text


def chose_search(mode, call, body):
    """n63 10-15: a protocol-v2 search call is ~51 tokens (median) > max_new, so its JSON is cut off before it closes
    and parse_response returns call None; the decision is the chosen tool name, which comes in the first ~15 tokens."""
    return mode == tool_data.MODE_TOOL and ((bool(call) and call.get("name") == "search")
                                            or '"name": "search"' in (body or ""))


known = []
for i, r in enumerate(ans):
    q, gold = r["prompt"].strip(), r["chosen"].strip().lower()
    forced = decode(q, no_search, True)
    k = gold in forced.lower()
    s = chose_search(*tool_data.parse_response(decode(q, names, False)))
    known.append((k, s))
    if i < 3:
        print(f"  {q[:60]!r} gold={gold!r} forced={forced[:50]!r} known={k} search={s}", flush=True)
nk = sum(k for k, _ in known); nu = len(known) - nk
ps_k = sum(s for k, s in known if k) / max(1, nk); ps_u = sum(s for k, s in known if not k) / max(1, nu)
un_ok = 0
for r in unans:
    mode, call, body = tool_data.parse_response(decode(r["prompt"].strip(), names, False))
    un_ok += chose_search(mode, call, body) or \
             (mode == tool_data.MODE_ANSWER and any(w in (body or "").lower() for w in IDK))
print(f"selfknow {a.run}: really known {nk}/{len(known)} ({nk / max(1, len(known)):.1%}) | P(search|known) {ps_k:.3f} | "
      f"P(search|unknown) {ps_u:.3f} | selectivity {ps_u - ps_k:+.3f} | unanswerable -> search or IDK "
      f"{un_ok / max(1, len(unans)):.3f} (n={len(unans)})", flush=True)
