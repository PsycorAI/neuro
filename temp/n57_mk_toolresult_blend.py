"""n57_mk_toolresult_blend.py -- 2026-10-12 (Neuro 57, plan D3). Assemble the second-pass SFT file from the raw teacher
output, applying the fixes found on 10-12:
  * search: the questions written FROM pack passages (toolresult_search.jsonl) + at most 15% "I could not find that"
    lines (from toolresult.jsonl, where the result really was "no result found"); the other search lines are dropped
    (live search failed for most calls during generation, so their replies are "could not" for the wrong reason)
  * send_message: the reply must name the recipient;  any tool: no "could not" reply when the tool succeeded
Out: training_data/posttrain/toolresult_blend.jsonl ({instruction, input, output})      python temp/n57_mk_toolresult_blend.py"""
import json, re, random, collections
PT = "/home/glenn/projects/neuro/training_data/posttrain/"
rng = random.Random(0)
out, cnf, c = [], [], collections.Counter()
for l in open(PT + "toolresult.jsonl", encoding="utf-8"):
    r = json.loads(l); cn = "could not" in r["output"].lower()
    if r["tool"] == "search":
        if cn and "no result found" in r["input"]:
            cnf.append(r)
        continue
    if cn and r["ok"]:
        continue
    if r["tool"] == "send_message":
        m = re.search(r"message to (.+?) placed", r["input"])
        if not m or m.group(1).lower() not in r["output"].lower():
            continue
    out.append(r)
sq = [json.loads(l) for l in open(PT + "toolresult_search.jsonl", encoding="utf-8")]
rng.shuffle(cnf)
out += sq + cnf[:int(len(sq) * 0.15 / 0.85)]
rng.shuffle(out)
with open(PT + "toolresult_blend.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        c[r["tool"]] += 1
        f.write(json.dumps(dict(instruction=r["instruction"], input=r["input"], output=r["output"]), ensure_ascii=False) + "\n")
print(f"[OK] toolresult_blend.jsonl: {len(out)} lines", dict(c))
