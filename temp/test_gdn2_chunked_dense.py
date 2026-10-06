"""Parity: gdn2_chunked_dense (WY-GEMM, dense state) vs gdn2_sequential (oracle).
Cases: small exact, real scale w/ nonzero M0, extreme decay, ragged T, backward grads."""
import torch, sys
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from delta_chunked import gdn2_sequential, gdn2_chunked_dense

torch.manual_seed(0)
dev = "cuda"
fails = []

def make(B, T, D, alpha_lo=0.3, alpha_hi=0.9999, grad=False):
    k = torch.randn(B, T, D, device=dev)
    v = torch.randn(B, T, D, device=dev)
    q = torch.randn(B, T, D, device=dev)
    b = torch.rand(B, T, D, device=dev)
    w = 0.1 + 0.9 * torch.rand(B, T, D, device=dev)
    alpha = alpha_lo + (alpha_hi - alpha_lo) * torch.rand(B, T, D, device=dev)
    M0 = torch.randn(B, D, D, device=dev) * 0.1
    ts = [k, v, q, b, w, alpha]
    if grad:
        for t in ts:
            t.requires_grad_(True)
    return ts + [M0]

def check(tag, B, T, D, chunk=64, **kw):
    k, v, q, b, w, alpha, M0 = make(B, T, D, **kw)
    kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    r_ref, M_ref = gdn2_sequential(v, kn, q, b, w, alpha, M0.clone())
    r_new, M_new = gdn2_chunked_dense(v, k, q, b, w, alpha, M0.clone(), chunk=chunk)
    r_err = (r_ref - r_new).abs().max().item()
    r_rel = r_err / (r_ref.abs().max().item() + 1e-9)
    cos = torch.nn.functional.cosine_similarity(r_ref.reshape(-1), r_new.reshape(-1), dim=0).item()
    M_err = (M_ref - M_new).abs().max().item()
    ok = r_rel < 2e-3 and cos > 0.99999
    print(f"[{tag}] r max_err={r_err:.2e} rel={r_rel:.2e} cos={cos:.7f} M_err={M_err:.2e} {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(tag)

check("small D=64 T=128", 2, 128, 64)
check("real D=1024 T=512", 2, 512, 1024)
check("extreme decay 0.01-0.999", 2, 256, 128, alpha_lo=0.01, alpha_hi=0.999)
check("trained-regime decay 1e-6", 2, 512, 256, alpha_lo=1e-6, alpha_hi=0.9999)
check("mixed dead+live channels", 4, 512, 1024, alpha_lo=1e-6, alpha_hi=0.5)
check("ragged T=100 chunk=64", 2, 100, 64)
check("chunk=32", 2, 128, 64, chunk=32)

# correlated-keys stress (MQAR-like: few distinct keys repeated many times,
# high erase gates -> the regime that blew up unit_lower_inv in real training)
kpool = torch.randn(2, 8, 64, device=dev)
k = kpool[:, torch.randint(0, 8, (512,), device=dev)]
v = torch.randn(2, 512, 64, device=dev)
q = kpool[:, torch.randint(0, 8, (512,), device=dev)]
b = 0.90 + 0.099 * torch.rand(2, 512, 64, device=dev)
w = 0.90 + 0.099 * torch.rand(2, 512, 64, device=dev)
alpha = 0.95 + 0.0499 * torch.rand(2, 512, 64, device=dev)
M0 = torch.zeros(2, 64, 64, device=dev)
kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
r_ref, M_ref = gdn2_sequential(v, kn, q, b, w, alpha, M0.clone())
r_new, M_new = gdn2_chunked_dense(v, k, q, b, w, alpha, M0.clone())
r_rel = (r_ref - r_new).abs().max().item() / (r_ref.abs().max().item() + 1e-9)
finite = torch.isfinite(r_new).all().item() and torch.isfinite(M_new).all().item()
print(f'[correlated-keys stress] rel={r_rel:.2e} finite={finite} ' + ('OK' if r_rel < 2e-3 and finite else 'FAIL'))
if not (r_rel < 2e-3 and finite):
    fails.append('correlated-keys')

# backward grads
k, v, q, b, w, alpha, M0 = make(2, 128, 64, grad=True)
kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
r_ref, _ = gdn2_sequential(v, kn, q, b, w, alpha, M0.clone())
r_ref.pow(2).mean().backward()
g_ref = [t.grad.clone() for t in (k, v, q, b, w, alpha)]
for t in (k, v, q, b, w, alpha):
    t.grad = None
r_new, _ = gdn2_chunked_dense(v, k, q, b, w, alpha, M0.clone())
r_new.pow(2).mean().backward()
worst = 0
for n, gr, t in zip("kvqbwa", g_ref, (k, v, q, b, w, alpha)):
    rel = (gr - t.grad).abs().max().item() / (gr.abs().max().item() + 1e-9)
    worst = max(worst, rel)
    print(f"  grad[{n}] rel={rel:.2e}")
if worst > 5e-3:
    fails.append("backward")
print("ALL PASS" if not fails else f"FAILURES: {fails}")
