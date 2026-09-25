"""Autonomous decision script: given the two 30M β-fix logs, decide whether
either fix is "promising" enough to warrant the 9h 90M seed1 validation run.

Decision criteria (rationale in the recommendation given to Glenn):
  Baseline: 30M seed0 MoR final memory-Δ = +2.35 (near-collapsed).
  We need clear improvement over that to justify 9h of 90M validation.

  PROMISING = (a) mean memory-Δ across last-3 evals >= +10 (≈4x baseline)
              AND (b) at least one block with dead_frac <= 30% at final eval.

  If BOTH fixes are promising: prefer beta_floor (cheaper, no code path change,
  the chunked/FLA kernel still applies).
  If ONLY ONE is promising: pick that one.
  If NEITHER: don't launch 90M -- log the finding, wait for human review.

Output: writes /tmp/beta_fix_decision.txt with one of:
  LAUNCH beta_floor
  LAUNCH vector_beta
  SKIP <reason>
Exit code 0 always (so the watchdog reads the file rather than gating on
$?, which composes cleanly with error handling).
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path("/home/glenn/projects/neuro")
DECISION_FILE = "/tmp/beta_fix_decision.txt"
BASELINE_MEMORY_DELTA = 2.35  # 30M seed0 MoR final

FIXES = [
    ("beta_floor",  ROOT / "logs_screen_mor_beta_floor_30m.txt"),
    ("vector_beta", ROOT / "logs_screen_mor_vector_beta_30m.txt"),
]

# Regexes on the training log lines emitted by train_gpu.py evaluate()
EVAL_RE = re.compile(r"\[eval\]\s+val_ppl\s+([\d.]+)\s+\|\s+ablated_ppl\s+([\d.]+)\s+"
                     r"\(memory\s+.=([+\-]?[\d.]+)\)")
BETA_RE = re.compile(r"\[beta\]\s+(.+)")
BLK_RE  = re.compile(r"blk(\d+):\s+.=([\d.]+)\s+dead=([\d.]+)%")


def parse_log(path):
    """Return dict with 'deltas' (list of memory-Δ per eval, in order) and
    'dead_fracs' (list of per-eval [blk_dead_frac_list] in order)."""
    if not path.exists():
        return None
    deltas = []
    dead_fracs = []
    with open(path) as f:
        for line in f:
            m = EVAL_RE.search(line)
            if m:
                deltas.append(float(m.group(3)))
            m = BETA_RE.search(line)
            if m:
                block_hits = BLK_RE.findall(m.group(1))
                dead_fracs.append([float(d) / 100.0 for _, _, d in block_hits])
    return {"deltas": deltas, "dead_fracs": dead_fracs}


def is_promising(name, data):
    """Apply the promising criteria; return (bool, human_reason_str)."""
    if data is None:
        return False, "log missing"
    if not data["deltas"]:
        return False, "no eval lines yet"
    last3 = data["deltas"][-3:]
    mean_delta = sum(last3) / len(last3)
    ok_delta = mean_delta >= 10.0
    if data["dead_fracs"]:
        last_dead = data["dead_fracs"][-1]
        min_dead = min(last_dead) if last_dead else 1.0
        n_healthy = sum(1 for d in last_dead if d <= 0.30)
    else:
        min_dead = None
        n_healthy = 0
    ok_dead = n_healthy >= 1
    reason = (f"mean memory-Δ (last 3 evals) = {mean_delta:+.2f} "
              f"[need ≥ +10.0] ⇒ {'PASS' if ok_delta else 'FAIL'}; "
              f"blocks with dead_frac≤30% = {n_healthy} "
              f"[need ≥1] ⇒ {'PASS' if ok_dead else 'FAIL'}")
    return (ok_delta and ok_dead), reason


def main():
    results = {}
    print("=" * 60)
    print("β-fix result analysis")
    print("=" * 60)
    print(f"Baseline (30M seed0 MoR) final memory-Δ = +{BASELINE_MEMORY_DELTA}")
    print()
    for name, path in FIXES:
        data = parse_log(path)
        promising, reason = is_promising(name, data)
        results[name] = (promising, reason, data)
        n_evals = len(data["deltas"]) if data else 0
        final_delta = data["deltas"][-1] if data and data["deltas"] else None
        print(f"[{name}] ({n_evals} eval lines, final memory-Δ = "
              f"{final_delta:+.2f} vs baseline {BASELINE_MEMORY_DELTA:+.2f})"
              if final_delta is not None else
              f"[{name}] no data")
        print(f"  {reason}")
        print(f"  PROMISING: {promising}")
        print()

    bf_ok = results["beta_floor"][0]
    vb_ok = results["vector_beta"][0]

    if bf_ok and vb_ok:
        decision = "LAUNCH beta_floor"
        why = "both fixes promising; picking beta_floor (cheaper, same code path)"
    elif bf_ok:
        decision = "LAUNCH beta_floor"
        why = "only beta_floor promising"
    elif vb_ok:
        decision = "LAUNCH vector_beta"
        why = "only vector_beta promising"
    else:
        decision = "SKIP neither promising"
        why = "neither fix cleared the promising bar; hold 90M validation for review"

    print("=" * 60)
    print(f"DECISION: {decision}")
    print(f"WHY: {why}")
    print("=" * 60)

    with open(DECISION_FILE, "w") as f:
        f.write(decision + "\n")
    print(f"wrote {DECISION_FILE}")


if __name__ == "__main__":
    main()
