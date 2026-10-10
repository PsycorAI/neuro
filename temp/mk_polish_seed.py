"""mk_polish_seed.py -- 2026-10-06 (Neuro 53; Glenn's idea: accept the ppl cost of memory training, recover it at the end).
Seed a late "polish" run from a trained ckpt: same weights, NEW config, no optimizer state; the trainer then continues
the schedule like a resume (as temp/mk_spike_seed.py does for conversions).
  python temp/mk_polish_seed.py SRC.pt DST_DIR CONFIG.yaml"""
import sys, os, torch, yaml
src, dst_dir, cfg = sys.argv[1:4]
s = torch.load(src, map_location="cpu", weights_only=False)
out = {k: v for k, v in s.items() if k not in ("opt_states", "opt", "cfg")}
out["cfg"] = yaml.safe_load(open(cfg))
os.makedirs(dst_dir, exist_ok=True)
torch.save(out, os.path.join(dst_dir, f"step_{s['step']:07d}.pt"))
print(f"[OK] polish seed {dst_dir} @ step {s['step']} from {src}")
