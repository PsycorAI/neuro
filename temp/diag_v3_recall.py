# diag_v3_recall.py -- why does flagship_v3 score ~50% @N=4 on recall_eval? (2026-09-26)
# Checks (a) missing keys silently ignored by strict=False, (b) replay None vs split.
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import recall_eval as R

run = sys.argv[1] if len(sys.argv) > 1 else "flagship_v3_363m"
ck = R.latest(run); print("ckpt", ck)
blob = torch.load(ck, map_location="cuda", weights_only=False)
c = blob["cfg"]
print("cfg keys of interest:", {k: c.get(k) for k in ["attn_ratio", "attn_every", "n_attn", "block_pattern", "vocab", "mqar_weight", "mqar_range", "gate_rank", "attn_single_residual"]})
m = R.build(c, "cuda")
miss, unexp = m.load_state_dict(blob["model"], strict=False)
print(f"missing={len(miss)} unexpected={len(unexp)}")
print("missing sample:", miss[:8]); print("unexpected sample:", unexp[:8])
for rp in [None, "split"]:
    row = []
    for N in [4, 8, 16, 32]:
        _, r, ca = R.recall(m, c, N, "cuda", B=32, replay=rp)
        row.append(f"N{N}={ca*100:.0f}%/r{r:.0f}")
    print(f"replay={rp}:", " ".join(row))
