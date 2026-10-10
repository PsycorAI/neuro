"""chk_valppl_only.py -- 2026-10-09 (Neuro 54). watchdog_valppl_only: trips with watchdog_enabled false; stays silent
without the flag; and the v4_cloud tuning does not trip on r26n's real val series."""
import sys
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import _valppl_watchdog_step as W


def run(series, c):
    st = {"valppl_hits": 0, "best_valppl": None, "best_step": None, "valppl_ema": None}
    return any(W(st, v, 1000 * (i + 1), c)[1] for i, v in enumerate(series))


V4 = dict(watchdog_enabled=False, watchdog_valppl_only=True, watchdog_valppl_ema_alpha=0.4, watchdog_valppl_tol=0.07,
          watchdog_valppl_patience=2)
div = [60, 50, 45, 42, 41, 44, 48, 53, 60, 70]
r26n = [146.24, 91.21, 73.38, 84.41, 73.35, 59.79, 51.42, 46.90, 45.44, 42.21, 40.12, 38.72]
assert run(div, V4), "must trip on a divergence"
assert not run(div, dict(V4, watchdog_valppl_only=False)), "must stay silent when both flags are off"
assert not run(r26n, V4), "must not trip on r26n"
print("[OK] watchdog_valppl_only: trips on divergence, silent when off, silent on r26n")
