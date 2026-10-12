"""selfknow_probe.py -- 2026-10-12 (Neuro 57, docs/demo_v4_plan_2026_10_12.md D4; R-Tuning style). Which questions
does THIS checkpoint answer correctly? Then write the "knows what it does not know" SFT records for it.
Probe: 4-shot "Question: / Answer:" prompt, the gold answer teacher-forced (one batched forward per question).
  known    every gold token is the model's top choice
  unknown  the first gold token is not in the model's top 5         (the band in between is left out)
Pool: training_data/posttrain/abstain.jsonl (6,400 answerable with gold, 9,600 unanswerable = unknown by construction).
Records (tool records, all hash splits; tool_data.load_tool_jsonl picks train / test):
  sk_known   ANSWER <gold>                          (search is offered 70% of the time: do not search what you know)
  sk_search  TOOL search {"query": <question>}       (unknown, search offered)
  sk_idk     ANSWER I don't know. (5 wordings)       (unknown, search NOT offered)
Out: training_data/posttrain/selfknow_<run>.jsonl + a summary line.
  CUDA_VISIBLE_DEVICES=1 python scripts/selfknow_probe.py --run r26o_13L_RC3_conv_llamakd_150m"""
import os, sys, glob, json, argparse, random
import torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from train_gpu import build_model
from train_sft import make_encoder_pair
PT = os.path.join(ROOT, "training_data", "posttrain")
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--cap-search", type=int, default=4500)
ap.add_argument("--cap-idk", type=int, default=2500)
a = ap.parse_args()
rng = random.Random(0)
SHOTS = ("Question: What is the capital of France?\nAnswer: Paris\n\nQuestion: How many days are in a week?\nAnswer: seven\n\n"
         "Question: What gas do plants take in from the air?\nAnswer: carbon dioxide\n\n"
         "Question: Which planet is known as the Red Planet?\nAnswer: Mars\n\n")
IDK = ["I don't know.", "I don't know that.", "I do not know.", "I don't know the answer to that.",
       "I'm not sure about that; I don't know."]
NON_SEARCH = ["calculator", "unit_convert", "get_weather", "get_time", "define", "translate", "set_reminder", "send_message"]

ans, unans = [], []
for l in open(os.path.join(PT, "abstain.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    (ans if r["pair_type"] == "answerable" else unans).append(r)
ck = sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", a.run, "step_*.pt")))[-1]
blob = torch.load(ck, map_location="cuda", weights_only=False); c = dict(blob["cfg"])
m = build_model(c).to("cuda"); m.load_state_dict(blob["model"], strict=False); m.eval()
encode, _ = make_encoder_pair(dict(c, lut_subword_fallback=True))

label = {}
items = []
for r in ans:
    p = encode(SHOTS + f"Question: {r['prompt'].strip()}\nAnswer:")
    g = encode(" " + r["chosen"].strip())
    if g and len(p) + len(g) < c["block_size"]:
        items.append((r, p, g))
items.sort(key=lambda t: len(t[1]) + len(t[2]))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i in range(0, len(items), a.batch):
        ch = items[i:i + a.batch]
        T = max(len(p) + len(g) for _, p, g in ch)
        x = torch.zeros(len(ch), T, dtype=torch.long, device="cuda")
        for j, (_, p, g) in enumerate(ch):
            x[j, :len(p) + len(g)] = torch.tensor(p + g)
        top = m(x).float().topk(5, -1).indices            # right padding cannot change earlier positions (causal)
        for j, (r, p, g) in enumerate(ch):
            pred = top[j, len(p) - 1:len(p) + len(g) - 1]
            gt = torch.tensor(g, device="cuda")
            if bool((pred[:, 0] == gt).all()):
                label[r["prompt"]] = "known"
            elif not bool((pred[0] == gt[0]).any()):
                label[r["prompt"]] = "unknown"
n_known = sum(v == "known" for v in label.values()); n_unk = sum(v == "unknown" for v in label.values())
print(f"probe {a.run}: {len(items)} answerable -> known {n_known} ({n_known / len(items):.1%}), unknown {n_unk} "
      f"({n_unk / len(items):.1%}), band {len(items) - n_known - n_unk}", flush=True)


def offered(with_search):
    t = rng.sample(NON_SEARCH, rng.choice([2, 3, 4, 5])) + (["search"] if with_search else [])
    rng.shuffle(t)
    return t


recs = []
for r in ans:
    if label.get(r["prompt"]) == "known":
        recs.append(dict(kind="sk_known", instruction=r["prompt"].strip(), tools=offered(rng.random() < 0.7),
                         target_mode="ANSWER", target_text=r["chosen"].strip()))
unk = [r for r in ans if label.get(r["prompt"]) == "unknown"]
rng.shuffle(unk); rng.shuffle(unans)
half_s, half_i = a.cap_search // 2, a.cap_idk // 2
pool_s = unk[:half_s] + unans[:a.cap_search - min(half_s, len(unk))]
pool_i = unk[half_s:half_s + half_i] + unans[a.cap_search:a.cap_search + a.cap_idk - min(half_i, max(0, len(unk) - half_s))]
for r in pool_s:
    q = r["prompt"].strip()
    recs.append(dict(kind="sk_search", instruction=q, tools=offered(True), target_mode="TOOL",
                     target_call=dict(name="search", arguments=dict(query=q.rstrip("?").strip()))))
for r in pool_i:
    recs.append(dict(kind="sk_idk", instruction=r["prompt"].strip(), tools=offered(False), target_mode="ANSWER",
                     target_text=rng.choice(IDK)))
rng.shuffle(recs)
out = os.path.join(PT, f"selfknow_{a.run}.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
kinds = {k: sum(r["kind"] == k for r in recs) for k in ("sk_known", "sk_search", "sk_idk")}
print(f"[OK] {out}: {len(recs)} records {kinds}", flush=True)
