"""test_lif_exact.py -- correctness (vs float64 autograd reference) + speed vs lif_parallel."""
import sys, time, math, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from lif_exact import lif_exact
import lif_parallel as LP
torch.set_float32_matmul_precision("high")
dev = "cuda"; torch.manual_seed(0)

def ref(cur, beta, vth=1.0, alpha=2.0):
    """float64: true sequential spikes; then the SAME surrogate graph as lif_parallel's final scan."""
    B, T, N = cur.shape
    with torch.no_grad():
        mem = torch.zeros(B, N, dtype=cur.dtype, device=cur.device); ss = []
        for t in range(T):
            pre = beta * mem + cur[:, t]; s = (pre > vth).to(cur.dtype); ss.append(s); mem = pre - s * vth
        s = torch.stack(ss, 1)
    s_prev = torch.zeros_like(s); s_prev[:, 1:] = s[:, :-1]
    x = cur - beta * vth * s_prev
    y = torch.zeros(B, N, dtype=cur.dtype, device=cur.device); pres = []
    for t in range(T):
        y = beta * y + x[:, t]; pres.append(y)
    pre = torch.stack(pres, 1)
    over = pre - vth; hard = (over > 0).to(cur.dtype)
    g = (1.0 / (math.pi * alpha)) * torch.atan(math.pi * alpha * over)
    return hard.detach() + g - g.detach(), pre[:, -1] - hard[:, -1] * vth

ok = True
for (B, T, N, beta) in [(2, 64, 300, 0.9), (3, 128, 513, 0.95), (2, 512, 257, 0.9)]:
    c64 = (torch.randn(B, T, N, device=dev, dtype=torch.float64) * 0.6 + 0.1).requires_grad_()
    c32 = c64.detach().float().requires_grad_()
    gs = torch.randn(B, T, N, device=dev, dtype=torch.float64); gm = torch.randn(B, N, device=dev, dtype=torch.float64)
    s_r, m_r = ref(c64, beta); (s_r * gs).sum().add((m_r * gm).sum()).backward()
    s_k, m_k = lif_exact(c32, beta); (s_k * gs.float()).sum().add((m_k * gm.float()).sum()).backward()
    flips = (s_k.detach().double().round() != s_r.detach().round()).double().mean().item()
    em = (m_k.double() - m_r).abs().max().item()
    eg = ((c32.grad.double() - c64.grad).abs().max() / c64.grad.abs().max()).item()
    good = flips < 1e-4 and em < 1e-4 and eg < 1e-4
    ok &= good
    print(f"B{B} T{T} N{N} beta{beta}: spike flips {flips:.1e} | mem_final err {em:.1e} | grad rel err {eg:.1e} -> {'PASS' if good else 'FAIL'}")
# bf16 input path runs and matches the fp32 kernel on the same (rounded) input
cb = (torch.randn(4, 512, 1000, device=dev) * 0.6 + 0.1).to(torch.bfloat16)
sb, mb = lif_exact(cb, 0.9); sf, mf = lif_exact(cb.float(), 0.9)
print(f"bf16 path: dtype {sb.dtype}, spikes identical to fp32-on-same-input: {torch.equal(sb.float(), sf)}")
print("ALL PASS" if ok else "SOME FAIL")

def timeit(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3
for N in (3840, 9728):
    cur = (torch.randn(8, 512, N, device=dev) * 0.6 + 0.1).to(torch.bfloat16).requires_grad_()
    for name, f in [("lif_parallel K10", lambda c: LP.lif_parallel(c, 0.9, K=10)), ("lif_exact", lambda c: lif_exact(c, 0.9))]:
        def fb():
            s, m = f(cur); (s.float().sum() + m.float().sum()).backward()
        print(f"B8 T512 N{N} {name:17s}: fwd {timeit(lambda: f(cur)):6.2f} ms | fwd+bwd {timeit(fb):6.2f} ms")
