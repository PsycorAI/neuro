"""Equivalence test: lif_parallel (FPT) vs the sequential _lif_step in model.py.

Run on GPU 1 (5080) so it doesn't compete with the 5090 training run:
    CUDA_VISIBLE_DEVICES=1 python tests/test_fpt_equivalence.py
"""
import math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
from lif_parallel import lif_parallel, _scan_chunked, _scan_sequential


def sequential_lif(cur, beta=0.9, vth=1.0, alpha=2.0):
    """Match SpikingHebbianBlock._lif_step exactly, looped over time."""
    B, T, N = cur.shape
    device, dtype = cur.device, cur.dtype
    mem = torch.zeros(B, N, device=device, dtype=dtype)
    spks = torch.empty_like(cur)
    for t in range(T):
        mem = beta * mem + cur[:, t, :]
        over = mem - vth
        spk_hard = (over > 0).to(dtype)
        g = (1.0 / (math.pi * alpha)) * torch.atan(math.pi * alpha * over)
        spks[:, t, :] = spk_hard.detach() + g - g.detach()
        mem = mem - spk_hard * vth
    return spks, mem


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  ({torch.cuda.get_device_name(0)})")

    torch.manual_seed(0)
    # Test 1: scan equivalence
    B, T, N = 4, 128, 64
    x = torch.randn(B, T, N, device=device) * 0.5
    beta = 0.9
    y_seq = _scan_sequential(beta, x)
    y_par = _scan_chunked(beta, x.float()).to(x.dtype)
    err = (y_seq - y_par).abs().max().item()
    print(f"[scan] max abs error: {err:.2e}")
    assert err < 1e-4, f"scan mismatch: {err}"
    print("[OK] parallel scan matches sequential")

    # Test 2: LIF forward equivalence vs K
    print("\nSpike-pattern match vs K (T=128):")
    torch.manual_seed(42)
    cur128 = torch.randn(B, 128, N, device=device) * 0.6
    spk_seq, _ = sequential_lif(cur128, beta=0.9)
    spk_seq_hard = (spk_seq.detach() > 0.5).to(torch.float32)
    for K in [1, 3, 5, 10, 20]:
        spk_par, _ = lif_parallel(cur128, beta=0.9, K=K)
        spk_par_hard = (spk_par.detach() > 0.5).to(torch.float32)
        m = (spk_seq_hard == spk_par_hard).float().mean().item()
        print(f"  K={K:2d}: {m*100:.3f}% match")

    print("\nSpike-pattern match vs T (K=10):")
    for T in [16, 64, 128, 256]:
        torch.manual_seed(42)
        cur = torch.randn(B, T, N, device=device) * 0.6
        spk_seq, mem_seq = sequential_lif(cur, beta=0.9)
        spk_par, mem_par = lif_parallel(cur, beta=0.9, K=10)
        # check spike binary patterns
        spk_seq_hard = (spk_seq.detach() > 0.5).to(torch.float32)
        spk_par_hard = (spk_par.detach() > 0.5).to(torch.float32)
        match = (spk_seq_hard == spk_par_hard).float().mean().item()
        sr = spk_seq_hard.mean().item()
        print(f"[T={T:4d}] spike-pattern match {match*100:.2f}%  spike_rate {sr:.3f}")

    # Test 3: speed comparison
    print("\nSpeed test (B=8, N=2048, varied T):")
    for T in [64, 128, 256, 512]:
        cur = torch.randn(8, T, 2048, device=device) * 0.5
        # warmup
        for _ in range(3):
            sequential_lif(cur)
            lif_parallel(cur, beta=0.9, K=3)
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.time()
        for _ in range(5):
            sequential_lif(cur)
        torch.cuda.synchronize() if device == "cuda" else None
        t_seq = (time.time() - t0) / 5
        t0 = time.time()
        for _ in range(5):
            lif_parallel(cur, beta=0.9, K=3)
        torch.cuda.synchronize() if device == "cuda" else None
        t_par = (time.time() - t0) / 5
        print(f"  T={T:4d}  seq {t_seq*1000:7.2f}ms  par {t_par*1000:7.2f}ms  "
              f"speedup {t_seq/t_par:.2f}x")

    # Test 4: gradient flow
    cur_base = (torch.randn(2, 32, 16, device=device) * 0.5)
    cur = cur_base.clone().detach().requires_grad_(True)
    spk, _ = lif_parallel(cur, beta=0.9, K=3)
    spk.sum().backward()
    g = cur.grad.abs().mean().item()
    print(f"\n[grad] mean |grad| through FPT: {g:.4f} (should be > 0)")
    assert g > 0, "no gradient flowed"
    print("[OK] gradient flows through parallel LIF")


if __name__ == "__main__":
    main()
