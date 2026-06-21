"""Equivalence + speed test for chunkwise Hebbian vs sequential."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
from hebbian_chunked import hebbian_chunked


def sequential_hebbian(v_seq, k_seq, q_seq, M_init, lam, eta):
    """Reference: per-timestep loop."""
    B, T, D = v_seq.shape
    M = M_init.clone()
    r_out = torch.empty(B, T, D, device=v_seq.device, dtype=v_seq.dtype)
    for t in range(T):
        M = lam * M + eta * torch.bmm(v_seq[:, t, :].unsqueeze(2),
                                       k_seq[:, t, :].unsqueeze(1))
        r_out[:, t, :] = torch.bmm(M, q_seq[:, t, :].unsqueeze(2)).squeeze(2)
    return r_out, M


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  ({torch.cuda.get_device_name(0)})")
    torch.manual_seed(0)

    # Test 1: equivalence (small)
    B, T, D = 4, 64, 32
    v = torch.randn(B, T, D, device=device) * 0.1
    k = torch.randn(B, T, D, device=device) * 0.1
    q = torch.randn(B, T, D, device=device) * 0.1
    M0 = torch.randn(B, D, D, device=device) * 0.01
    lam, eta = 0.98, 1.0

    r_seq, Mf_seq = sequential_hebbian(v, k, q, M0, lam, eta)
    r_chunk, Mf_chunk = hebbian_chunked(v, k, q, M0, lam, eta, chunk=16)
    err_r = (r_seq - r_chunk).abs().max().item()
    err_M = (Mf_seq - Mf_chunk).abs().max().item()
    print(f"[equiv] max abs error  r: {err_r:.2e}   M: {err_M:.2e}")
    assert err_r < 1e-3, f"r mismatch: {err_r}"
    assert err_M < 1e-3, f"M mismatch: {err_M}"
    print("[OK] chunkwise matches sequential (D=32, T=64)")

    # Test 2: vary chunk size — all should match
    print("\nChunk-size equivalence (T=128, D=64):")
    B, T, D = 4, 128, 64
    v = torch.randn(B, T, D, device=device) * 0.1
    k = torch.randn(B, T, D, device=device) * 0.1
    q = torch.randn(B, T, D, device=device) * 0.1
    M0 = torch.zeros(B, D, D, device=device)
    r_seq, Mf_seq = sequential_hebbian(v, k, q, M0, lam, eta)
    for chunk in [1, 8, 32, 64, 128]:
        r_c, Mf_c = hebbian_chunked(v, k, q, M0, lam, eta, chunk=chunk)
        e = (r_seq - r_c).abs().max().item()
        print(f"  chunk={chunk:4d}: max abs error {e:.2e}")

    # Test 3: speed (production scale)
    print("\nSpeed (B=16, D=1024, varied T) on full Hebbian path:")
    B, D = 16, 1024
    for T in [64, 128, 256, 512]:
        v = torch.randn(B, T, D, device=device) * 0.1
        k = torch.randn(B, T, D, device=device) * 0.1
        q = torch.randn(B, T, D, device=device) * 0.1
        M0 = torch.zeros(B, D, D, device=device)
        # warmup
        for _ in range(2):
            sequential_hebbian(v, k, q, M0, lam, eta)
            hebbian_chunked(v, k, q, M0, lam, eta, chunk=64)
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            sequential_hebbian(v, k, q, M0, lam, eta)
        if device == "cuda": torch.cuda.synchronize()
        t_seq = (time.time() - t0) / 3
        t0 = time.time()
        for _ in range(3):
            hebbian_chunked(v, k, q, M0, lam, eta, chunk=64)
        if device == "cuda": torch.cuda.synchronize()
        t_ch = (time.time() - t0) / 3
        print(f"  T={T:4d}  seq {t_seq*1000:7.2f}ms  chunked {t_ch*1000:7.2f}ms  "
              f"speedup {t_seq/t_ch:.2f}x")

    # Test 4: gradient flow
    v = torch.randn(2, 32, 16, device=device, requires_grad=True) * 0.1
    k = torch.randn(2, 32, 16, device=device, requires_grad=True) * 0.1
    q = torch.randn(2, 32, 16, device=device, requires_grad=True) * 0.1
    M0 = torch.zeros(2, 16, 16, device=device)
    r, _ = hebbian_chunked(v, k, q, M0, lam, eta, chunk=8)
    r.sum().backward()
    print(f"\n[grad] |grad_v|={v.grad.abs().mean():.4f}  |grad_k|={k.grad.abs().mean():.4f}  "
          f"|grad_q|={q.grad.abs().mean():.4f}")
    assert v.grad.abs().sum() > 0 and k.grad.abs().sum() > 0 and q.grad.abs().sum() > 0
    print("[OK] gradients flow through chunked Hebbian")


if __name__ == "__main__":
    main()
