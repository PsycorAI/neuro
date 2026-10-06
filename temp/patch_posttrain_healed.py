"""patch_posttrain_healed.py -- 2026-09-28 (Glenn: finish v3 for a demo): point the v3 post-training pipeline at the HEALED
base (heal_v3_363m = flagship_v3 10B ckpt + 1B honest tokens after the bf16 leak fix) instead of the leak-trained 10B ckpt.
Also tightens heal ckpts (4000 -> 2000 steps) and gives the 426M transformer v3's retuned watchdog (EMA 0.4, tol 0.07)."""
N = "/home/glenn/projects/neuro/"
def edit(path, pairs):
    s = open(N + path, encoding="utf-8", newline="").read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a, s.count(a)); s = s.replace(a, b)
    open(N + path, "w", encoding="utf-8", newline="").write(s); print("[OK]", path)
edit("configs/sft_v3_363m.yaml", [("sft_init: flagship_v3_363m",
     "sft_init: heal_v3_363m    # 2026-09-28: HEALED base (was flagship_v3_363m = leak-trained 10B ckpt)")])
edit("temp/run_v3_posttrain.sh", [
     ('                       "flagship_v3_363m/step_*.pt"))', '                       "heal_v3_363m/step_*.pt"))   # 2026-09-28: healed base'),
     ("MIN_TOKENS=${MIN_TOKENS:-9500000000}", "MIN_TOKENS=${MIN_TOKENS:-10900000000}   # healed base = 11B (2026-09-28)"),
     ('--runs flagship_v3_363m,"$run"', '--runs heal_v3_363m,"$run"')])
edit("configs/heal_v3_363m.yaml", [("ckpt_every: 4000", "ckpt_every: 2000        # 2026-09-28: tighter (was 4000)")])
edit("configs/baseline_tfmr426_L17.yaml", [("watchdog_valppl_tol: 0.02",
     "watchdog_valppl_tol: 0.07          # = v3's retuned watchdog (the default 0.02 false-tripped v3 on plateau noise)\nwatchdog_valppl_ema_alpha: 0.4")])
