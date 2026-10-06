# flops_2L_3L.py -- 2026-09-30 CPU-only: training FLOPs/token (fwd+bwd, FlopCounterMode) for 2L vs 3L signed + r13 variants.
# Counts matmul-class ops only (elementwise/scan ops are memory-bound and not counted). B=1, T=512, no MQAR aux.
import sys, yaml, torch
from torch.utils.flop_counter import FlopCounterMode
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model
C = "/home/glenn/projects/neuro/configs/"
runs = ["screen_r5_signed_s0_30m", "screen_r8_signed_3L_s0_30m", "screen_r13a_signed_3L_attn2_s0_30m",
        "screen_r13d_signed_n4096_s0_30m", "screen_tfmr3_adamw_s0_30m"]
T = 512
torch.manual_seed(0)
for r in runs:
    try:
        c = yaml.safe_load(open(C + r + ".yaml"))
    except FileNotFoundError:
        print(r, "config missing"); continue
    m = build_model(c).cuda()
    x = torch.randint(0, c["vocab"], (1, T)).cuda()
    fc = FlopCounterMode(display=False)
    with fc:
        out = m(x)
        out = out[0] if isinstance(out, tuple) else out
        out.float().mean().backward()
    tot = fc.get_total_flops() / T
    blk = {k: v for k, v in fc.get_flop_counts().items() if k.count(".") <= 2}
    print(f"{r:40s} {tot / 1e6:8.1f} MFLOP/tok (train)  params {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M", flush=True)
    del m
