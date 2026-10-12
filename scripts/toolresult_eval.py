"""toolresult_eval.py -- 2026-10-12 (Neuro 57, docs/demo_v4_plan_2026_10_12.md D3). Second pass of the tool loop:
given the question and `### Input: Result from <tool>: <text>`, does the model's reply carry the result?
Held-out data: training_data/posttrain/toolresult_test.jsonl (+ toolresult_search_test.jsonl).
  faithful   exact tools (calculator, unit_convert, get_time, get_weather, translate): the result's key value is in
             the reply.   search: >= 60% of the gold answer's content words are in the reply.
             others (define, reminder, message): >= 50% of the gold reply's content words.
  python scripts/toolresult_eval.py --run sft_v4_reh13_v2tr [--n 40]      (n = examples per tool)"""
import os, sys, glob, json, re, argparse, random
import torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from train_gpu import build_model
from train_sft import make_encoder_pair
from sft_data import PROMPT_WITH_INPUT, EOS_MARK
PT = os.path.join(ROOT, "training_data", "posttrain")
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--max-new", type=int, default=60)
a = ap.parse_args()
STOP = set("the a an of in on at to and or is are was were by for with as it its that which from this be has had have i you your".split())


def words(s):
    return [w for w in re.findall(r"[a-z0-9.]+", s.lower()) if w not in STOP]


def key_value(tool, res):
    if tool == "calculator":
        return res
    if tool == "unit_convert" and "=" in res:
        return res.split("=")[1].split()[0]
    if tool == "get_time":
        return res.split(" on ")[0].split()[0]
    if tool == "get_weather":
        m = re.search(r"(-?\d+) degrees", res)
        return m.group(1) if m else None
    if tool == "translate":
        return res.rstrip(".!?")
    return None


rows = {}
for fn in ("toolresult_test.jsonl", "toolresult_search_test.jsonl"):
    for l in open(os.path.join(PT, fn), encoding="utf-8"):
        r = json.loads(l)
        if fn == "toolresult_test.jsonl" and (r["tool"] == "search" or not r.get("ok", True)):
            continue
        rows.setdefault(r["tool"], []).append(r)
ck = sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", a.run, "step_*.pt")))[-1]
blob = torch.load(ck, map_location="cuda", weights_only=False); c = blob["cfg"]
m = build_model(c).to("cuda"); m.load_state_dict(blob["model"], strict=False); m.eval()
encode, decode = make_encoder_pair(c)
tot = hit = 0
for tool, rs in sorted(rows.items()):
    random.Random(0).shuffle(rs); rs = rs[:a.n]; h = 0
    for i, r in enumerate(rs):
        ids = encode(PROMPT_WITH_INPUT.format(instruction=r["instruction"].strip(), input=r["input"].strip()))
        out = []
        with torch.no_grad():
            for _ in range(a.max_new):
                tok = int(m(torch.tensor([(ids + out)[-c["block_size"]:]], device="cuda"))[0, -1].argmax())
                out.append(tok)
                if EOS_MARK in decode(out[-4:]):
                    break
        text = decode(out).split(EOS_MARK)[0].strip()
        res = r["input"].split(": ", 1)[1]
        kv = key_value(tool, res)
        if kv is not None:
            ok = kv.lower() in text.lower() or kv.replace(",", "") in text.replace(",", "")
        else:
            g = words(r["output"]); tw = set(words(text))
            ok = bool(g) and sum(w in tw for w in g) / len(g) >= (0.6 if tool == "search" else 0.5)
        h += ok
        if i < 1:
            print(f"    [{tool}] {r['instruction'][:50]!r} | {res[:50]!r} -> {text[:90]!r}", flush=True)
    print(f"  toolresult {a.run} {tool:13s} faithful {h / len(rs):.3f} (n={len(rs)})", flush=True)
    tot += len(rs); hit += h
print(f"  toolresult {a.run} ALL faithful {hit / tot:.3f} (n={tot})")
