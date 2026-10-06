"""mk_insert_ckpt.py -- 2026-10-05 (r25l). Seed a run with a trained ckpt plus NEW blocks inserted at given depths:
old block i -> new index (the i-th index not in --insert); inserted blocks keep their fresh init. Spiking blocks loaded
from the ckpt get ls_gamma = 1 (the config turns spk_layerscale on so the NEW block starts near-silent at its init value).
Optimizer state is dropped (fresh optimizer); step/tokens kept so the run continues the source schedule like a resume.
  python temp/mk_insert_ckpt.py SRC.pt DST_DIR CONFIG.yaml"""
import sys, os, torch, yaml
src, dst_dir, cfg = sys.argv[1:4]
c = yaml.safe_load(open(cfg))
L, attn = c["n_layers"], set(c["attention_layer_positions"])
s = torch.load(src, map_location="cpu", weights_only=False)
old_L = s["cfg"]["n_layers"]
ins = [int(i) for i in c["insert_blocks"]]                # config key (ignored by the trainer)
keep = [i for i in range(L) if i not in ins]
assert len(keep) == old_L, (keep, old_L)
sd = {}
for k, v in s["model"].items():
    if k.startswith("blocks."):
        p = k.split(".")
        p[1] = str(keep[int(p[1])])
        k = ".".join(p)
    sd[k] = v
for i in keep:                                   # loaded spiking blocks: identity layer scale
    if i not in attn:
        sd[f"blocks.{i}.ls_gamma"] = torch.ones(c["d_mem"])
out = {k: v for k, v in s.items() if k not in ("opt_states", "opt", "model")}
out["model"] = sd
out["cfg"] = c
os.makedirs(dst_dir, exist_ok=True)
torch.save(out, os.path.join(dst_dir, f"step_{s['step']:07d}.pt"))
print(f"[OK] {src} -> {dst_dir}: inserted blocks {ins}, kept {len(keep)}; step {s['step']}")
