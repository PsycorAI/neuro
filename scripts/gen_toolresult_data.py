"""gen_toolresult_data.py -- 2026-10-12 (Neuro 57, docs/demo_v4_plan_2026_10_12.md D3). "Tool result -> final answer"
SFT data for the second pass of the tool loop, in the ordinary instruction format (no new syntax):

    ### Instruction: {the user's question}   ### Input: Result from calculator: 40.8   ### Response: {reply}

For up to --per-tool gold calls per tool from the TRAIN split of tools.jsonl: phase 1 runs the REAL executor
(scripts/demo/tools_exec.py; search / weather live with the disk cache; 15% of searches are replaced by "no result
found" so the model learns the honest failure reply) and stores the results (resumable); phase 2 has the teacher write
a one-to-two sentence reply that uses only the result. Filter: the reply must contain the result's key value.
Out: training_data/posttrain/toolresult.jsonl ({instruction, input, output, tool}); the held-out split is by the same
instruction hash as tools.jsonl (tool_data.split_of), which scripts using this file must respect.
  CUDA_VISIBLE_DEVICES=1 python scripts/gen_toolresult_data.py --per-tool 1000 --batch 48"""
import os, sys, json, re, time, random, argparse
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts", "demo"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import tool_data, tools_exec
PT = os.path.join(ROOT, "training_data", "posttrain")
ap = argparse.ArgumentParser()
ap.add_argument("--per-tool", type=int, default=1000)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--split", default="train", help="train -> toolresult.jsonl (SFT data); test -> toolresult_test.jsonl (eval)")
ap.add_argument("--exec-only", action="store_true")
a = ap.parse_args()
rng = random.Random(0)
tools = tool_data.load_tools()
SFX = "" if a.split == "train" else "_" + a.split
EXEC = os.path.join(PT, f"toolresult_exec{SFX}.jsonl")
OUT = os.path.join(PT, f"toolresult{SFX}.jsonl")

# ---------------------------------------------------------------- phase 1: run the executors (resumable)
done = {}
if os.path.exists(EXEC):
    for line in open(EXEC, encoding="utf-8"):
        r = json.loads(line); done[r["instruction"]] = r
pairs, _ = tool_data.load_tool_jsonl(os.path.join(PT, "tools.jsonl"), tools)
by_tool = {}
for p in pairs:
    rec = p["rec"]
    if rec["target_mode"] == "TOOL" and tool_data.split_of(rec["instruction"]) == a.split:
        by_tool.setdefault(rec["target_call"]["name"], []).append(rec)
memo = {}
with open(EXEC, "a", encoding="utf-8") as f:
    for name, recs in by_tool.items():
        rng.shuffle(recs)
        n_new = 0
        for rec in recs[:a.per_tool]:
            if rec["instruction"] in done:
                continue
            call = rec["target_call"]
            key = json.dumps(call, sort_keys=True)
            if name == "search" and rng.random() < 0.15:
                res = dict(ok=False, text="no result found", source="forced")
            else:
                if key not in memo:
                    memo[key] = tools_exec.run(name, call["arguments"])
                    if name in ("search", "get_weather") and memo[key]["source"] == "live":
                        time.sleep(0.25)                     # polite to the public APIs
                res = memo[key]
            row = dict(instruction=rec["instruction"], tool=name, arguments=call["arguments"], ok=res["ok"],
                       result=res["text"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            done[rec["instruction"]] = row; n_new += 1
        print(f"  exec {name}: +{n_new}", flush=True)
rows = list(done.values())
print(f"phase 1: {len(rows)} results, ok share {sum(r['ok'] for r in rows) / max(1, len(rows)):.3f}", flush=True)
if a.exec_only:
    sys.exit(0)


# ---------------------------------------------------------------- phase 2: the teacher writes the reply
def key_values(r):
    """Strings of which at least one must appear in the reply (lower-cased); None = no check."""
    t = r["result"]
    if not r["ok"]:
        return None
    if r["tool"] == "calculator":
        return [t, t.replace(",", "")]
    if r["tool"] == "unit_convert":
        v = t.split("=")[1].split()[0]
        return [v, v.replace(",", "")]
    if r["tool"] == "get_time":
        return [t.split(" on ")[0].lower(), t.split(" on ")[0].split()[0]]
    if r["tool"] == "translate":
        return [t.lower().rstrip(".!?")]
    if r["tool"] == "get_weather":
        m = re.search(r"(-?\d+) degrees", t)
        return [m.group(1)] if m else None
    return None


SYSTEM = ("You write the assistant's final reply to a user. You are given the user's request and the result a tool "
          "returned. Reply in one or two plain sentences using ONLY the tool result: copy numbers, names, times and "
          "translated text exactly, add no other facts. If the result says nothing was found, something failed or is "
          "unknown, or it does not answer the request, say 'I could not find that.' or 'I could not do that.' "
          "Do not mention tools or results; just answer the user. Always write one natural, complete English "
          "sentence as a helpful assistant would say it (only translated text itself may be in another language). "
          "Examples: 'It is 5:01 AM on Monday in New York.' | '349.8 inches is 888.492 centimeters.' | "
          "'609 divided by 61 is 9.98361.' | 'Done, I set a reminder to back up the laptop tonight at 11pm.' | "
          "'Your message to Dad is in the outbox: can you grab milk on the way home.' | "
          "'\"Happy birthday\" in Spanish is \"Feliz cumpleaños\".'")
have = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding="utf-8"):
        have.add(json.loads(line)["instruction"])
todo = [r for r in rows if r["instruction"] not in have
        and not (r["tool"] == "translate" and not r["ok"])]       # unsupported language: nothing useful to teach
print(f"phase 2: {len(todo)} replies to write ({len(have)} already there)", flush=True)
if todo:
    from gen_teacher_data import resolve_teacher, load_teacher, batched_chat
    tid = resolve_teacher()
    model, tok = load_teacher(tid)
    print(f"teacher: {tid}", flush=True)
    prompts = [(i, f"User request: {r['instruction']}\nTool: {r['tool']}\nTool result: {r['result']}\n\nYour reply:")
               for i, r in enumerate(todo)]
    kept = dropped = 0
    with open(OUT, "a", encoding="utf-8") as f:
        for _, res in batched_chat(model, tok, prompts, max_new=80, temperature=0.4, batch_size=a.batch, system=SYSTEM):
            for i, text in res:
                r = todo[i]
                text = re.sub(r"\s+", " ", text).strip().strip('"')
                kv = key_values(r)
                foreign = r["tool"] != "translate" and sum(ord(ch) > 127 for ch in text) > 3
                if (not text or len(text) > 400 or len(text.split()) < 4 or foreign
                        or (kv and not any(k.lower() in text.lower() for k in kv))):
                    dropped += 1
                    continue
                f.write(json.dumps(dict(instruction=r["instruction"], input=f"Result from {r['tool']}: {r['result']}",
                                        output=text, tool=r["tool"], ok=r["ok"]), ensure_ascii=False) + "\n")
                kept += 1
            f.flush()
            if (kept + dropped) % (a.batch * 10) < a.batch:
                print(f"  {kept} kept, {dropped} dropped", flush=True)
    print(f"done: {kept} kept, {dropped} dropped by the key-value filter", flush=True)
