"""gen_searchqa_data.py -- 2026-10-12 (Neuro 57, plan D3 fix). Second-pass data for the SEARCH tool.
Why: with the tools.jsonl search questions ("tallest building finished this year") the Wikipedia summary almost never
holds the answer, so 78% of the teacher's replies were "I could not find that" -- training on that would teach the
model to give up after every search. Here the direction is reversed: for each search RESULT already fetched
(toolresult_exec*.jsonl), the teacher writes --k questions that the passage answers, each with a one-sentence answer
taken from the passage. Filter: the answer's content words must mostly occur in the passage.
Out: training_data/posttrain/toolresult_search{,_test}.jsonl, same record shape as toolresult.jsonl.
  CUDA_VISIBLE_DEVICES=1 python scripts/gen_searchqa_data.py --split train --k 3"""
import os, sys, json, re, argparse
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
PT = os.path.join(ROOT, "training_data", "posttrain")
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--pack", type=int, default=0, help="take this many random passages from the offline search pack")
a = ap.parse_args()
SFX = "" if a.split == "train" else "_" + a.split
if a.pack:      # passages straight from the offline search pack (what the demo's offline search returns)
    import sqlite3, random
    sys.path.insert(0, os.path.join(ROOT, "scripts", "demo"))
    import tools_exec
    con = sqlite3.connect(tools_exec.PACK)
    rows = con.execute("select title, body from docs where length(body) between 200 and 700").fetchall()
    random.Random(1 if a.split == "train" else 2).shuffle(rows)
    passages = [tools_exec._clip(f"{t}: {b}") for t, b in rows[:a.pack]]
else:
    passages = sorted({json.loads(l)["result"] for l in open(os.path.join(PT, f"toolresult_exec{SFX}.jsonl"), encoding="utf-8")
                       if '"tool": "search"' in l and json.loads(l)["ok"]})
print(f"{len(passages)} distinct search results", flush=True)
SYSTEM = (f"You are given a short encyclopedia passage. Write {a.k} different factual questions that a curious person "
          "might ask and that this passage answers, each followed by a one-sentence answer that uses only words and "
          "facts from the passage. The questions must name their subject (never say 'the passage' or 'this'). "
          "Format exactly:\nQ: <question>\nA: <answer>\n" + "Q: ...\nA: ...\n" * (a.k - 1))
from gen_teacher_data import resolve_teacher, load_teacher, batched_chat
model, tok = load_teacher(resolve_teacher())
STOP = set("the a an of in on at to and or is are was were by for with as it its that which from this be has had have".split())
kept = dropped = 0
with open(os.path.join(PT, f"toolresult_search{SFX}.jsonl"), "w", encoding="utf-8") as f:
    for _, res in batched_chat(model, tok, [(i, "Passage: " + p) for i, p in enumerate(passages)], max_new=200,
                               temperature=0.5, batch_size=a.batch, system=SYSTEM):
        for i, text in res:
            p = passages[i]; pw = set(re.findall(r"[a-z0-9]+", p.lower()))
            for q, ans in re.findall(r"Q:\s*(.+?)\s*\nA:\s*(.+?)\s*(?:\n|$)", text):
                aw = [w for w in re.findall(r"[a-z0-9]+", ans.lower()) if w not in STOP]
                if (not q.endswith("?") or len(aw) < 2 or len(ans) > 300 or "passage" in (q + ans).lower()
                        or sum(w in pw for w in aw) / len(aw) < 0.8):
                    dropped += 1
                    continue
                f.write(json.dumps(dict(instruction=q, input=f"Result from search: {p}", output=ans, tool="search",
                                        ok=True), ensure_ascii=False) + "\n")
                kept += 1
        f.flush()
print(f"done: {kept} kept, {dropped} dropped", flush=True)
