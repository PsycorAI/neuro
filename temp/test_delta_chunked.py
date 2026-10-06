"""Equivalence test: delta_chunked vs delta_sequential (the model.py step() math).

Passes if chunked == sequential to <1e-4 across chunk sizes, non-divisible T,
nonzero carry-in M, and lam in {0.98, 1.0}.
"""
import sys, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from delta_chunked import delta_sequential, delta_chunked

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
B, T, D = 3, 130, 2048        # D=2048 = flagship d_mem; T=130 exercises ragged last chunk

worst = 0.0
for lam in (0.98, 1.0):
    for zero_carry in (True, False):
        v = torch.randn(B, T, D, device=dev)
        k = torch.randn(B, T, D, device=dev)
        q = torch.randn(B, T, D, device=dev)
        beta = torch.sigmoid(torch.randn(B, T, 1, device=dev))
        M0 = torch.zeros(B, D, D, device=dev) if zero_carry else 0.1 * torch.randn(B, D, D, device=dev)

        r_seq, M_seq = delta_sequential(v, k, q, beta, M0, lam)
        for chunk in (1, 8, 16, 64, 256):     # 256 > T -> single chunk
            r_ch, M_ch = delta_chunked(v, k, q, beta, M0, lam, chunk=chunk)
            dr = (r_ch - r_seq).abs().max().item()
            dM = (M_ch - M_seq).abs().max().item()
            worst = max(worst, dr, dM)
            tag = "OK " if max(dr, dM) < 1e-4 else "!!!"
            print(f"[{tag}] lam={lam} zero_carry={zero_carry} chunk={chunk:3d} "
                  f"| max|dr|={dr:.2e} max|dM|={dM:.2e}")

print(f"\nWORST max-abs-diff across all cases: {worst:.2e}  "
      f"-> {'PASS' if worst < 1e-4 else 'FAIL'}")
