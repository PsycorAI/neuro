"""check_scan_leak.py -- does lif_parallel/_scan_chunked leak FUTURE inputs when x is bf16?"""
import sys, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import lif_parallel as LP
print("bf16 arange(512) exact?", torch.equal(torch.arange(512, dtype=torch.bfloat16).float(), torch.arange(512).float()),
      "| e.g. 257,259,261 ->", torch.tensor([257., 259., 261.]).to(torch.bfloat16).tolist())
torch.manual_seed(0)
for dt in (torch.float32, torch.bfloat16):
    x = (torch.randn(2, 512, 64, device="cuda") * 0.5).to(dt)
    y0 = LP._scan_chunked(0.9, x).float()
    leaks = []
    for p in (100, 255, 258, 260, 300, 400, 511):
        xp = x.clone(); xp[:, p] += 5.0
        d = (LP._scan_chunked(0.9, xp).float() - y0).abs().amax(dim=(0, 2))   # (T,)
        leaks.append(f"p={p}: max|dy| before p = {d[:p].max().item():.3g}")
    print(f"{dt}: " + " | ".join(leaks))
    # full neuron, spikes: does a future input change past spikes?
    cur = (torch.randn(2, 512, 256, device="cuda") * 0.6 + 0.1).to(dt)
    s0 = LP.lif_parallel(cur, 0.9, K=10)[0].round()
    cp = cur.clone(); cp[:, 300] += 3.0
    s1 = LP.lif_parallel(cp, 0.9, K=10)[0].round()
    print(f"   lif_parallel K=10 spikes changed BEFORE t=300 by a perturbation AT t=300: {(s1[:, :300] != s0[:, :300]).float().mean().item():.4f}")
