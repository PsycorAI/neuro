M = "/mnt/c/Users/Glenn/.claude/projects/C--Users-Glenn/memory/"
p = M + "neuro_ppl_gap_diagnosis_2026_09_27.md"
s = open(p).read()
anchor = "**How to apply:** follow `docs/ppl_gap_plan_2026_09_27.md`"
assert s.count(anchor) == 1
add = """**Round 10-13 (2026-09-30, Neuro 46) — details in docs/to-do.html entries of that date:**
- **signed+canon CONFIRMED 2-seed = 59.52** (59.74/59.29; 0.87x 2L tfmr; firing 12.5%) = best 30M stack.
- **Gentle gap curriculum WORKS** (r7_gapw015, unsigned stack, mqar gap440 at mqar_weight 0.15): 71.10 (no ppl cost vs ~74.9), haystack ~99-100% at all gaps, half-life ~3700 tok. r12 = signed+canon + same (running).
- **Signed sparsity Pareto** lambda 0.5/1/2/4 -> 61.24/62.73/64.60/67.46 @ firing 12.7/9.3/6.7/4.8%; recall survives everywhere.
- **DEPTH WARNING:** 2L->3L: tfmr -14%, unsigned -11%, signed +3-5% (worse) -> signed lead may vanish at v4 depth; 3L signed horizon collapse is seed-dependent; 3L = +32% FLOPs but 1.7x wall time (eager, GDN-2 chunk-loop launches). r13a/b/d depth screens queued. Judge depth/width at matched WALL-CLOCK.

"""
s = s.replace(anchor, add + anchor)
open(p, "w").write(s)
p = M + "reference_gpu_schedule.md"
s = open(p).read()
s = s.rstrip("\n") + ("\n\n**Update 2026-09-30:** Glenn sometimes uses the 5080 past midnight -> `temp/run_5080_sched.sh` now waits before EVERY "
     "start/resume/eval until GPU util <=5% for 5 consecutive min (`FREE_UTIL`/`FREE_MIN`; logs every 15 min while busy). "
     "Old version: `temp/run_5080_sched.sh.bak_20260930`.\n")
open(p, "w").write(s)
print("[OK]")
