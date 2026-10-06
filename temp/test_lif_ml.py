"""test_lif_ml.py -- 2026-09-28: lif_exact_ml (multi-level / signed / per-neuron learnable beta) vs a float64
sequential PyTorch reference with the same staircase atan surrogate and detached reset.
Checks: spikes exact, mem_final, grad wrt cur, grad wrt beta; and levels=1/unsigned/scalar beta == lif_exact."""
import sys, math, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from lif_exact import lif_exact, lif_exact_ml

def ref(cur, beta, vth, alpha, gamma, L, signed):
    B, T, N = cur.shape
    mem = torch.zeros(B, N, dtype=torch.float64, device=cur.device)
    pa = math.pi * alpha
    G = lambda o: torch.atan(pa * o) / pa
    out = []
    for t in range(T):
        pre = beta * mem + cur[:, t]
        pos = torch.clamp(torch.floor(pre / vth), 0, L)
        hard = pos - (torch.clamp(torch.floor(-pre / vth), 0, L) if signed else 0)
        g = sum(G(pre - j * vth) for j in range(1, L + 1))
        if signed:
            g = g - sum(G(-pre - j * vth) for j in range(1, L + 1))
        s = hard.detach() + g - g.detach()
        out.append(s)
        mem = pre - gamma * hard.detach() * vth
    return torch.stack(out, 1), mem

torch.manual_seed(0)
ok = True
for L, signed, learn in [(1, False, True), (4, False, False), (4, False, True), (1, True, True), (2, True, True)]:
    B, T, N = 3, 257, 300
    cur = (torch.randn(B, T, N, device="cuda", dtype=torch.float64) * 0.8 + 0.2)
    beta0 = torch.rand(N, device="cuda", dtype=torch.float64) * 0.45 + 0.5
    gw = torch.randn(B, T, N, device="cuda", dtype=torch.float64); gm = torch.randn(B, N, device="cuda", dtype=torch.float64)
    # reference
    c1 = cur.clone().requires_grad_(); b1 = beta0.clone().requires_grad_(learn)
    s1, m1 = ref(c1, b1, 1.0, 2.0, 1.0, L, signed)
    ((s1 * gw).sum() + (m1 * gm).sum()).backward()
    # kernel (fp32)
    c2 = cur.float().clone().requires_grad_(); b2 = beta0.float().clone().requires_grad_(learn)
    s2, m2 = lif_exact_ml(c2, b2, 1.0, alpha=2.0, gamma=1.0, levels=L, signed=signed)
    ((s2 * gw.float()).sum() + (m2 * gm.float()).sum()).backward()
    flips = (s1.detach().float() != s2.detach()).float().mean().item()
    em = (m1.detach().float() - m2.detach()).abs().max().item()
    ec = ((c1.grad.float() - c2.grad).abs().max() / c1.grad.abs().max()).item()
    eb = ((b1.grad.float() - b2.grad).abs().max() / b1.grad.abs().max()).item() if learn else 0.0
    good = flips < 1e-4 and em < 1e-3 and ec < 1e-4 and eb < 1e-4
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} L={L} signed={signed} learn_beta={learn}: spike flips {flips:.1e} | "
          f"mem_final err {em:.1e} | grad_cur rel {ec:.1e} | grad_beta rel {eb:.1e} | "
          f"mean|s| {s2.detach().abs().mean():.3f} max {s2.detach().abs().max():.0f}")
# levels=1 unsigned scalar beta: must equal the original lif_exact (forward + grad)
cur = torch.randn(2, 300, 500, device="cuda") * 0.8
a = cur.clone().requires_grad_(); b = cur.clone().requires_grad_()
sa, ma = lif_exact(a, 0.9); sb, mb = lif_exact_ml(b, 0.9)
g = torch.randn_like(sa); (sa * g).sum().backward(); (sb * g).sum().backward()
same = torch.equal(sa, sb) and torch.equal(ma, mb) and (a.grad - b.grad).abs().max().item() < 1e-6
ok &= same
print(f"{'PASS' if same else 'FAIL'} L=1 scalar beta == lif_exact (spikes/mem exact, grad max diff {(a.grad - b.grad).abs().max().item():.1e})")
print("ALL PASS" if ok else "TEST FAILED")
