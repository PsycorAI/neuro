"""bench_scan.py -- 2026-09-27: LIF scan lever microbench (5080).

Current _scan_chunked(chunk=512) at T=512 = ONE dense 512x512 fp32/TF32 matmul per scan.
Candidate: two-level scan with chunk c (intra-chunk c x c matmul, all chunks in ONE batched
matmul; inter-chunk carry via a tiny (nC x nC) matmul) -- same math, ~512/c fewer FLOPs,
no Python loop. Checks: time, max |diff| vs the float64 sequential reference, and spike
agreement of the full lif_parallel (K=10) on realistic currents.
"""
import sys, time, math, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import lif_parallel as LP
torch.set_float32_matmul_precision("high")   # same as train_gpu (TF32)
dev = "cuda"

def scan2(beta, x, c=64):
    B, T, N = x.shape
    assert T % c == 0
    nC = T // c
    idx = torch.arange(c, device=x.device, dtype=x.dtype)
    diff = idx[:, None] - idx[None, :]
    W = torch.where(diff >= 0, beta ** diff.clamp(min=0), torch.zeros_like(diff))   # (c,c)
    xc = x.view(B, nC, c, N)
    y = torch.matmul(W, xc)                                  # (B,nC,c,N) intra-chunk
    f = y[:, :, -1, :]                                       # chunk-final (B,nC,N)
    # carry into chunk k = sum_{j<k} beta^(c*(k-1-j)) f_j   (strictly lower triangular)
    k = torch.arange(nC, device=x.device, dtype=x.dtype)
    d = k[:, None] - 1 - k[None, :]
    Wc = torch.where(d >= 0, (beta ** c) ** d.clamp(min=0), torch.zeros_like(d))   # (nC,nC)
    carry = torch.einsum("kj,bjn->bkn", Wc, f)               # (B,nC,N)
    y = y + (beta ** (idx + 1)).view(1, 1, c, 1) * carry.unsqueeze(2)
    return y.view(B, T, N)

def timeit(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3

torch.manual_seed(0)
for (B, N) in [(8, 3840), (8, 9728)]:
    T = 512
    for beta in (0.9, 0.95):
        x = torch.randn(B, T, N, device=dev) * 0.5
        ref = LP._scan_sequential(beta, x.double())
        t_old = timeit(lambda: LP._scan_chunked(beta, x))
        e_old = (LP._scan_chunked(beta, x).double() - ref).abs().max().item()
        line = f"B{B} N{N} beta{beta}: old(512) {t_old:.2f}ms err {e_old:.1e}"
        for c in (32, 64, 128):
            t = timeit(lambda: scan2(beta, x, c))
            e = (scan2(beta, x, c).double() - ref).abs().max().item()
            line += f" | c{c} {t:.2f}ms ({t_old/t:.1f}x) err {e:.1e}"
        print(line, flush=True)

# spike agreement of the full FPT LIF (K=10), bf16-ish realistic currents
beta = 0.9
for scale in (0.3, 0.6, 1.0):
    cur = torch.randn(8, 512, 3840, device=dev) * scale + 0.1
    s_old, m_old = LP.lif_parallel(cur, beta, K=10)
    orig = LP._scan_chunked
    LP._scan_chunked = lambda b, x: scan2(b, x, 64)
    s_new, m_new = LP.lif_parallel(cur, beta, K=10)
    LP._scan_chunked = orig
    # sequential ground truth (float64)
    mem = torch.zeros(8, 3840, device=dev, dtype=torch.float64); spk = []
    for t in range(512):
        pre = beta * mem + cur[:, t].double()
        s = (pre > 1.0).double(); spk.append(s); mem = pre - s
    s_seq = torch.stack(spk, 1).float()
    fo = (s_old.detach().round() != s_seq).float().mean().item()
    fn_ = (s_new.detach().round() != s_seq).float().mean().item()
    print(f"cur scale {scale}: spike rate {s_seq.mean():.3f} | flips vs seq: old {fo:.2e} new(c64) {fn_:.2e}", flush=True)
