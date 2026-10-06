"""mk_gap_p0a_configs.py -- 2026-09-27: configs for ppl-gap plan P0 (tfmr LR check) and A1/A2 (spiking
optimizer). See docs/ppl_gap_plan_2026_09_27.md. Text edits of the control configs, one change each."""
import re
C = "/home/glenn/projects/neuro/configs/"
def mk(src, dst, header, subs):
    s = open(C + src, encoding="utf-8").read()
    body = "\n".join(l for l in s.splitlines() if not l.startswith("#")) + "\n"
    for k, v in subs.items():
        body, n = re.subn(rf"^{k}:.*$", f"{k}: {v}", body, flags=re.M)
        assert n == 1, (dst, k, n)
    open(C + dst, "w", encoding="utf-8", newline="").write(header + body)
    print("[OK]", dst)
H = "# {name} -- ppl-gap plan {tag} (2026-09-27, docs/ppl_gap_plan_2026_09_27.md). AUTO-GEN temp/mk_gap_p0a_configs.py\n# from {src}. ONE change: {what}.\n"
for lr, tag in (("0.0005", "lr5e4"), ("0.002", "lr2e3")):
    n = f"screen_tfmr_adamw_{tag}_s0_30m"
    mk("screen_tfmrfix_adamw_s0_30m.yaml", n + ".yaml",
       H.format(name=n, tag="P0", src="screen_tfmrfix_adamw_s0_30m (lr 1e-3, 68.5)", what=f"AdamW lr {lr}"),
       {"lr": lr, "run_name": n})
n = "screen_honest_mu02_s0_30m"
mk("screen_honest_lifexact_s0_30m.yaml", n + ".yaml",
   H.format(name=n, tag="A1", src="screen_honest_lifexact_s0_30m (NorMuon 0.005 / AdamW 6e-4, 99.3)",
            what="NorMuon muon_lr 0.02 + AdamW lr 1e-3 (tfmr's best Muon LR / its AdamW LR)"),
   {"muon_lr": "0.02", "lr": "0.001", "run_name": n})
n = "screen_honest_adamw_s0_30m"
mk("screen_honest_lifexact_s0_30m.yaml", n + ".yaml",
   H.format(name=n, tag="A2", src="screen_honest_lifexact_s0_30m (NorMuon 0.005 / AdamW 6e-4, 99.3)",
            what="optimizer adamw lr 1e-3 = the transformer's exact optimizer (NOTE: plain AdamW also decays 1D params)"),
   {"optimizer": "adamw", "lr": "0.001", "run_name": n})
