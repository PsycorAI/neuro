"""run_through.py -- 2026-10-14 (Neuro 61; demo plan D6 reliability). Drives the demo engine (scripts/demo/engine.py, no
HTTP) through scripted scenarios and checks each expected tool / reply. Ready = passes 3 times in a row, online and
offline. Edit SCENARIOS to the rehearsed demo questions.
  python scripts/demo/run_through.py --run <run> [--offline] [--device cuda] [--brain label=run ...]"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--device", default="cuda")
ap.add_argument("--offline", action="store_true"); ap.add_argument("--brain", action="append", default=[])
ap.add_argument("--stream", type=int, default=0)      # n78: W > 0 = engine_stream.StreamEngine(window=W)
ap.add_argument("--dtype", default="fp32")              # n78: StreamEngine precision (bf16 = the server default)
a = ap.parse_args()
from engine import Engine

# (question, expected tool or "ANSWER", substring the reply must contain or None)
SCENARIOS = [
    ("What is 17% of 240?", "calculator", "40.8"),
    ("How many kilometers is 26.2 miles?", "unit_convert", "42"),
    ("What time is it in Tokyo right now?", "get_time", None),
    ("What's the weather like in Paris?", "get_weather", None),
    ("What does the word 'ephemeral' mean?", "define", None),
    ("Translate 'good morning' into Spanish.", "translate", None),
    ("Remind me to call my son at 6 pm.", "set_reminder", None),
    ("Who wrote the novel Pride and Prejudice?", None, None),          # answer or search: both fine, must reply
]
if a.stream:
    from engine_stream import StreamEngine
    eng = StreamEngine(a.run, a.device, window=a.stream, dtype=a.dtype, brains=dict(b.split("=", 1) for b in a.brain))
else:
    eng = Engine(a.run, a.device, brains=dict(b.split("=", 1) for b in a.brain))
fails, t0 = 0, time.time()
for q, tool, must in SCENARIOS:
    eng.reset(); ev = []
    t = time.time()
    r = eng.assistant(q, ev.append, offline=a.offline)
    res = [e for e in ev if e["type"] == "tool_result"]
    ok = bool(r["reply"].strip())
    if tool == "ANSWER":
        ok &= r["mode"] == "ANSWER"
    elif tool:
        ok &= r["tool"] == tool and bool(res) and bool(res[0].get("ok"))
    if must:
        ok &= must in r["reply"]
    fails += not ok
    ntok = sum(1 for e in ev if e["type"] == "token" and e["kind"] == "gen")
    print(f"[{'OK' if ok else 'FAIL'}] {q!r} -> {r['mode']} {r['tool'] or ''} | {r['reply'][:80]!r} "
          f"({ntok} tok, {time.time() - t:.1f}s, {ntok / max(1e-6, time.time() - t):.1f} tok/s)", flush=True)
print(f"== run-through {a.run} {'offline' if a.offline else 'online'}: {len(SCENARIOS) - fails}/{len(SCENARIOS)} "
      f"{'PASSED' if not fails else 'FAILED'} in {time.time() - t0:.0f}s", flush=True)
sys.exit(1 if fails else 0)
