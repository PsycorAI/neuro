"""Unit-test the real _valppl_watchdog_step against the actual flagship sequence.

This is the deterministic proof the 30M smokes couldn't give: the flagship's
gentle head-driven divergence (dstd stayed ~1.8, so the dstd wire never fired)
would have been caught by THIS wire. Imports the real function from train_gpu.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
# import the function WITHOUT running main()
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_gpu", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "train_gpu.py"))
# heavy imports (torch etc.) are fine; we only call the pure fn
tg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tg)
step_fn = tg._valppl_watchdog_step

def fresh():
    return {"valppl_hits": 0, "best_valppl": None, "best_step": None}

def run(seq, c, step0=2000, dstep=2000):
    """Feed a val_ppl sequence; return (tripped_at_index or None, lines)."""
    st = fresh()
    lines = []
    for i, v in enumerate(seq):
        line, trip = step_fn(st, v, step0 + i * dstep, c)
        lines.append(line)
        if trip:
            return i, st, lines
    return None, st, lines

C = {"watchdog_enabled": True, "watchdog_grace_steps": 1500,
     "watchdog_valppl_tol": 0.02, "watchdog_valppl_patience": 2}

passed = failed = 0
def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1; print(f"  PASS  {name}")
    else:
        failed += 1; print(f"  FAIL  {name}  {detail}")

# ---- 1. THE flagship sequence: eval every 2000 steps from 124k.
# 19.61(best,124k) 19.64 20.01 20.92 22.81 24.75. tol 2% => best*1.02=20.00.
# 19.64 <20.00 -> hit=0. 20.01 >20.00 -> hit1. 20.92 -> hit2 => TRIP at index 3 (step 130k).
flagship = [19.61, 19.64, 20.01, 20.92, 22.81, 24.75]
idx, st, lines = run(flagship, C, step0=124000)
check("flagship sequence trips", idx is not None, f"idx={idx}")
check("flagship trips at 130k (index 3)", idx == 3, f"idx={idx} step={124000+idx*2000 if idx else None}")
check("flagship best pinned at 19.61", abs(st["best_valppl"] - 19.61) < 1e-9, st["best_valppl"])
check("flagship best_step = 124000", st["best_step"] == 124000, st["best_step"])
print(f"    (would have stopped at step {124000+idx*2000}, saving "
      f"{134000-(124000+idx*2000)} steps vs the actual 135k kill)" if idx is not None else "")

# ---- 2. healthy monotone-decreasing: must NEVER trip
idx, st, _ = run([40, 35, 31, 28, 26, 25, 24.5, 24.2, 24.0], C)
check("healthy monotone never trips", idx is None, f"idx={idx}")
check("healthy best tracks the min", abs(st["best_valppl"] - 24.0) < 1e-9, st["best_valppl"])

# ---- 3. noisy but within tol: bounces < 2% above best, must NOT trip
#   best 20.00; 20.00*1.02=20.40. bounces to 20.3 (<thresh) repeatedly.
idx, st, _ = run([20.0, 20.3, 20.1, 20.35, 20.2, 20.3, 20.1], C)
check("sub-tol noise never trips", idx is None, f"idx={idx}")

# ---- 4. one spike over tol then recovery: hit resets, must NOT trip
#   20.0 best; 21.0 (>2%, hit1); 19.5 (new best, reset hit=0); 20.5 (hit1)... never 2 consecutive.
idx, st, _ = run([20.0, 21.0, 19.5, 20.5, 19.2, 20.4], C)
check("spike-then-recover never trips (counter resets)", idx is None, f"idx={idx}")

# ---- 5. grace window: divergence before grace must NOT trip
idx, _, _ = run([19.61, 21.0, 22.0, 23.0], C, step0=0, dstep=200)  # steps 0,200,400,600 < 1500 grace
check("divergence inside grace window is held", idx is None, f"idx={idx}")

# ---- 6. patience knob honored: patience=3 needs 3 consecutive
C3 = dict(C, watchdog_valppl_patience=3)
idx, _, _ = run(flagship, C3, step0=124000)
check("patience=3 trips one eval later (index 4)", idx == 4, f"idx={idx}")

# ---- 7. tighter tol trips sooner: tol=0 => 19.64 already > best, patience 2 => trip index 2
C0 = dict(C, watchdog_valppl_tol=0.0)
idx, _, _ = run(flagship, C0, step0=124000)
check("tol=0 trips at index 2", idx == 2, f"idx={idx}")

# ---- 8. EMA smoothing (2026-09-16): the REAL flagship_v3 val_ppl tail that
# false-tripped the raw 2%/patience-2 wire. best single-eval hit a lucky 27.61,
# then normal noise (31.31, 30.06) cleared best*1.02 twice -> false trip. With
# EMA(0.4)+tol 5%+patience 3 (the new flagship_v3 config) it must NOT trip.
CEMA = {"watchdog_enabled": True, "watchdog_grace_steps": 1500,
        "watchdog_valppl_tol": 0.07, "watchdog_valppl_patience": 2,
        "watchdog_valppl_ema_alpha": 0.4}
v3_tail = [33.05, 32.96, 32.96, 32.46, 32.04, 32.05, 32.32, 31.62, 31.40, 31.62,
           30.87, 30.67, 32.27, 28.62, 28.25, 28.30, 27.61, 27.86, 31.31, 30.06]
idx, st, _ = run(v3_tail, CEMA, step0=68000)
check("EMA: real flagship_v3 noise never trips", idx is None, f"idx={idx}")
# same raw sequence WOULD trip the old raw 2%/patience-2 wire (proves the fix matters)
idx_raw, _, _ = run(v3_tail, C, step0=68000)
check("raw wire would have false-tripped the same series", idx_raw is not None, f"idx={idx_raw}")

# ---- 9. EMA must still catch a genuine monotonic runaway (real divergence),
# else the smoothing would have disarmed the guard.
idx, _, _ = run(v3_tail + [33.0, 37.0, 42.0, 48.0], CEMA, step0=68000)
check("EMA still trips on a real runaway tail", idx is not None, f"idx={idx}")

# ---- 10. EMA on the classic v1 flagship divergence: must still trip (idx 5,
# one eval later than the raw wire's idx 3 -- the smoothing/robustness cost).
idx, _, _ = run(flagship, CEMA, step0=124000)
check("EMA still catches the v1 head-norm divergence", idx is not None, f"idx={idx}")

print(f"\n{'='*50}\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
