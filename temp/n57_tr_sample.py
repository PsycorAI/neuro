"""n57_tr_sample.py -- Neuro 57. Per-tool counts, "could not" share and 2 samples of toolresult.jsonl."""
import json, collections
c, cn, s = collections.Counter(), collections.Counter(), {}
for l in open("/home/glenn/projects/neuro/training_data/posttrain/toolresult.jsonl", encoding="utf-8"):
    r = json.loads(l); c[r["tool"]] += 1; cn[r["tool"]] += "could not" in r["output"].lower()
    s.setdefault(r["tool"], []).append(r)
for t in c:
    print(f"{t}: {c[t]} lines, could-not {cn[t] / c[t]:.2f}")
    for r in s[t][3:5]:
        print("   ", r["input"][len("Result from " + t) + 2:][:70], "->", r["output"][:110])
