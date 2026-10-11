"""mk_kd_control_seed.py -- 2026-10-11 (Neuro 55). Seed for the conversion CONTROL r26y: the headline "conversion with
Llama KD = 36.55 < dense 38.09" compares 280M tokens + a teacher against 200M tokens without one. r26y = the same
+80M tokens, same Llama KD, same batch split and LR profile as r26o, but NO spike codes -> what the dense model reaches
with the same extra training. Conversion cost = r26o (36.55) - r26y.
Seed = r26n's final checkpoint without its optimizer state (r26o also starts its optimizer fresh).
  python temp/mk_kd_control_seed.py   -> models/checkpoints/r26y_13L_RC3_kdonly_150m/step_0012208.pt"""
import os, glob, torch
R = "/home/glenn/projects/neuro/models/checkpoints/"
src = sorted(glob.glob(R + "r26n_13L_RC2_xlate_150m/step_*.pt"))[-1]
dst = R + "r26y_13L_RC3_kdonly_150m/"
os.makedirs(dst, exist_ok=True)
s = torch.load(src, map_location="cpu", weights_only=False)
out = {k: v for k, v in s.items() if k not in ("opt_states", "opt")}
torch.save(out, dst + os.path.basename(src))
print(f"[OK] {dst + os.path.basename(src)}: step {out['step']}, {out['tokens'] / 1e6:.1f}M tok, keys {sorted(out)}")
