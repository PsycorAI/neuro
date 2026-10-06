"""smoke_gap_hybrid.py -- 2026-09-27: build the A3 hybrid + A2 AdamW configs, 3 train steps each on GPU (tiny batch)."""
import sys, yaml, torch, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, _build_optimizers
for name in ("screen_honest_hybrid_s0_30m", "screen_honest_adamw_s0_30m", "screen_honest_mu02_s0_30m"):
    c = yaml.safe_load(open(f"/home/glenn/projects/neuro/configs/{name}.yaml"))
    m = build_model(c).to("cuda"); opts = _build_optimizers(m, c, "cuda")
    kinds = [type(b).__name__ for b in m.blocks]
    x = torch.randint(0, c["vocab"], (2, 512), device="cuda")
    for i in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, sr = m(x[:, :-1], return_stats=True)
            loss = F.cross_entropy(lg.float().reshape(-1, c["vocab"]), x[:, 1:].reshape(-1))
        loss.backward(); [o.step() for o in opts]; [o.zero_grad() for o in opts]
    print(f"[OK] {name}: {sum(p.numel() for p in m.parameters())/1e6:.2f}M params | blocks {kinds} | "
          f"final_norm {m.out_norm is not None} | loss {loss.item():.3f} | spike_rate {sr.item():.3f}")
    del m, opts; torch.cuda.empty_cache()
