"""bench_lif.py -- 2026-09-27: lif_parallel variants, isolated (5080), bf16 currents as under autocast.
v0 = current lif_parallel (K=10, torch.equal early-exit = 1 host sync per iteration)
v1 = same math, no early-exit sync, F.pad shift (EXACT: past a fixed point every iteration reproduces s)
v2 = v1 with the no-grad fixed-point loop torch.compile'd
Reports ms fwd / fwd+bwd and exact spike equality vs v0.
"""
import sys, time, math, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import lif_parallel as LP
torch.set_float32_matmul_precision("high")
dev = "cuda"

def _fp_loop(cur, beta, vth, K):
    s = torch.zeros_like(cur)
    for _ in range(K):
        x = cur - beta * vth * F.pad(s[:, :-1, :], (0, 0, 1, 0))
        s = (LP._scan_chunked(beta, x) > vth).to(cur.dtype)
    return s
_fp_loop_c = torch.compile(_fp_loop, dynamic=False)

def lif_v1(cur, beta, vth=1.0, K=10, alpha=2.0, loop=_fp_loop):
    with torch.no_grad():
        s = loop(cur, beta, vth, K)
    x = cur - beta * vth * F.pad(s[:, :-1, :], (0, 0, 1, 0))
    pre_mem = LP._scan_chunked(beta, x)
    over = pre_mem - vth
    spk_hard = (over > 0).to(cur.dtype)
    g = (1.0 / (math.pi * alpha)) * torch.atan(math.pi * alpha * over)
    spk = spk_hard.detach() + g - g.detach()
    return spk, pre_mem[:, -1, :] - spk_hard[:, -1, :] * vth

def timeit(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3

torch.manual_seed(0)
for N in (3840, 9728):
    cur = (torch.randn(8, 512, N, device=dev) * 0.6 + 0.1).to(torch.bfloat16).requires_grad_()
    variants = {"v0": lambda c: LP.lif_parallel(c, 0.9, K=10),
                "v1": lambda c: lif_v1(c, 0.9, K=10),
                "v2": lambda c: lif_v1(c, 0.9, K=10, loop=_fp_loop_c),
                "v1_K3": lambda c: lif_v1(c, 0.9, K=3),
                "v1_K5": lambda c: lif_v1(c, 0.9, K=5)}
    s0, m0 = variants["v0"](cur)
    for name, f in variants.items():
        tf = timeit(lambda: f(cur))
        def fb():
            s, m = f(cur); (s.float().sum() + m.float().sum()).backward()
        tb = timeit(fb)
        s, m = f(cur)
        same = torch.equal(s.detach(), s0.detach()) and torch.equal(m, m0)
        flips = (s.detach() != s0.detach()).float().mean().item()
        print(f"N{N} {name:6s}: fwd {tf:6.2f} ms | fwd+bwd {tb:6.2f} ms | identical to v0: {same} (flip frac {flips:.2e})", flush=True)
