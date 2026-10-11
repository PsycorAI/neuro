"""sim_tripwire.py -- 2026-10-11 (Neuro 55; Glenn: make sure the averaged ppl monitor is what the H200 run uses and that
noise cannot stop the run). Runs the trainer's own decision function (train_gpu._valppl_watchdog_step, the EMA version)
over val curves at the CLOUD eval density (one eval per 3000 steps, ~75-100 per run) built from the real r26n curve:
  noise     the smooth curve + 2% random eval noise                    -> must never trip
  bump K    + the mid-run "recall emerges" bump seen in every dense run (r26n: 73.4 -> 84.4 -> 73.4, +15%; r26p +18.5%),
            lasting K consecutive evals                                 -> should not trip for any plausible K
  runaway   the flagship-v1 failure: val climbs 1.5% per eval, forever  -> must trip, and soon
for the current setting (EMA 0.4, tol 0.07, patience 2) and candidates. Pure CPU, a few ms."""
import sys, math, random
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import _valppl_watchdog_step

REAL = [146.24, 91.21, 73.38, 73.35, 59.79, 51.42, 46.90, 45.44, 42.21, 40.12, 38.72]   # r26n without its bump eval
N = 90                                                                                  # evals in a cloud dense run


def smooth(i):                       # log-linear interpolation of the real curve onto N evals
    x = i / (N - 1) * (len(REAL) - 1); a = int(x); b = min(a + 1, len(REAL) - 1)
    return math.exp(math.log(REAL[a]) + (x - a) * (math.log(REAL[b]) - math.log(REAL[a])))


def curve(kind, K=0, amp=0.15, seed=0):
    rnd = random.Random(seed); v = []
    for i in range(N):
        y = smooth(i) * (1 + rnd.gauss(0, 0.02))
        if kind == "bump" and 25 <= i < 25 + K:
            y *= 1 + amp
        if kind == "runaway" and i >= 60:
            y = smooth(60) * 1.015 ** (i - 60)
        v.append(y)
    return v


def trips(v, alpha, tol, pat):
    st = {"valppl_hits": 0, "best_valppl": None, "best_step": None, "valppl_ema": None}
    c = dict(watchdog_enabled=False, watchdog_valppl_only=True, watchdog_valppl_ema_alpha=alpha, watchdog_valppl_tol=tol,
             watchdog_valppl_patience=pat)
    for i, y in enumerate(v):
        _, t = _valppl_watchdog_step(st, y, (i + 1) * 3000, c)
        if t:
            return i
    return None


print(f"{'setting':28s} {'noise x200':>10s} " + " ".join(f"{'bump' + str(k):>7s}" for k in (1, 2, 3, 4, 6, 10)) + f" {'bump3 +18.5%':>13s} {'runaway: evals to trip':>23s}")
for alpha, tol, pat in ((0.4, 0.07, 2), (0.4, 0.07, 3), (0.4, 0.10, 3), (0.4, 0.15, 3), (0.4, 0.15, 4), (0.4, 0.20, 3)):
    nz = sum(trips(curve("noise", seed=s), alpha, tol, pat) is not None for s in range(200))
    bm = ["trip" if any(trips(curve("bump", K, seed=s), alpha, tol, pat) is not None for s in range(20)) else "ok" for K in (1, 2, 3, 4, 6, 10)]
    b18 = "trip" if any(trips(curve("bump", 3, 0.185, seed=s), alpha, tol, pat) is not None for s in range(20)) else "ok"
    ra = [trips(curve("runaway", seed=s), alpha, tol, pat) for s in range(20)]
    ra = [r - 60 for r in ra if r is not None]
    print(f"EMA {alpha} tol {tol:.2f} patience {pat}   {nz:>6d}/200 " + " ".join(f"{b:>7s}" for b in bm) + f" {b18:>13s} "
          f"{(str(min(ra)) + '-' + str(max(ra)) + ' evals, val +' + str(round((1.015 ** max(ra) - 1) * 100)) + '%') if len(ra) == 20 else 'NOT ALWAYS':>23s}")
