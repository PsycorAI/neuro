"""Ragged-tail padding fix (2026-09-25): gdn2_chunked_dense at chunk=256 with
T % 256 != 0 must match gdn2_sequential (fwd + grads), and peak memory must drop
vs the old ragged path. Run on GPU: CUDA_VISIBLE_DEVICES=0 venv python this."""
import sys, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from delta_chunked import gdn2_sequential, gdn2_chunked_dense
torch.manual_seed(0)
dev = "cuda"
ok = True

def case(B, T, D, chunk):
    global ok
    ts = [torch.randn(B, T, D, device=dev) for _ in range(3)]
    b = torch.rand(B, T, D, device=dev); w = 0.1 + 0.9 * torch.rand(B, T, D, device=dev)
    a = 0.3 + 0.6999 * torch.rand(B, T, D, device=dev)
    M0 = torch.randn(B, D, D, device=dev) * 0.1
    ins = [t.clone().requires_grad_(True) for t in (*ts, b, w, a)]
    k, v, q, bb, ww, aa = ins
    kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    r_ref, M_ref = gdn2_sequential(v, k, q, bb, ww, aa, M0)
    g_ref = torch.autograd.grad(r_ref.sum() + M_ref.sum(), ins)
    r, M = gdn2_chunked_dense(v, k, q, bb, ww, aa, M0, chunk=chunk)
    g = torch.autograd.grad(r.sum() + M.sum(), ins)
    err = max((r - r_ref).abs().max().item(), (M - M_ref).abs().max().item())
    gerr = max((x - y).abs().max().item() / (y.abs().max().item() + 1e-6) for x, y in zip(g, g_ref))
    good = err < 1e-3 and gerr < 1e-3
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} B={B} T={T} D={D} chunk={chunk}: fwd max|err|={err:.2e} grad rel err={gerr:.2e}")

for T in (300, 271, 257, 17, 511, 512):
    case(2, T, 64, 256)
case(2, 300, 256, 256)

# peak memory at SFT-like scale, one layer fwd+bwd, T=511 (tail 255)
B, T, D = 4, 511, 1280
x = [torch.randn(B, T, D, device=dev, requires_grad=True) for _ in range(6)]
M0 = torch.zeros(B, D, D, device=dev)
torch.cuda.reset_peak_memory_stats()
r, M = gdn2_chunked_dense(x[1], x[0], x[2], torch.sigmoid(x[3]), torch.sigmoid(x[4]),
                          0.5 + 0.49 * torch.sigmoid(x[5]), M0, chunk=256)
(r.sum() + M.sum()).backward()
print(f"peak mem B=4 T=511 D=1280 one layer fwd+bwd: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
print("ALL PASS" if ok else "SOME FAILED")
