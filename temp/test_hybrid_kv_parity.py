"""Parity test for HybridAttentionBlock's incremental KV-cache path.

Success criterion: step_kv (prefill + token-by-token) reproduces the full
forward_sequence output to <1e-4. Validates the new inference primitive before
it gets wired into model.forward / batched_rollout.
"""
import sys, torch
sys.path.insert(0, "/home/glenn/projects/neuro")
from src.hybrid_attn import HybridAttentionBlock

torch.manual_seed(0)
B, T, D, H = 2, 7, 64, 4
blk = HybridAttentionBlock(d_in=D, d_mem=D, n_neurons=32, n_heads=H,
                           use_rmsnorm=True, use_swiglu=True, use_qk_norm=True).eval()
x = torch.randn(B, T, D)

# dummy pass-through state slots (attention ignores them)
mem = torch.zeros(B, 32); M = torch.zeros(B, 1, 1); prev_spk = torch.zeros(B, 32)

with torch.no_grad():
    z_full, *_ = blk.forward_sequence(x, mem, M, prev_spk)

    # (a) pure token-by-token
    cache = None; zs = []
    for i in range(T):
        z_i, cache = blk.step_kv(x[:, i:i+1], cache)
        zs.append(z_i)
    z_step = torch.cat(zs, dim=1)

    # (b) chunk prefill (first 3) then step the rest -- exercises t>1 with cache
    cache = None
    z_pre, cache = blk.step_kv(x[:, :3], cache)
    zs2 = [z_pre]
    for i in range(3, T):
        z_i, cache = blk.step_kv(x[:, i:i+1], cache)
        zs2.append(z_i)
    z_mixed = torch.cat(zs2, dim=1)

d_step = (z_full - z_step).abs().max().item()
d_mixed = (z_full - z_mixed).abs().max().item()
print(f"max|full - token_by_token| = {d_step:.3e}")
print(f"max|full - chunk_prefill+step| = {d_mixed:.3e}")
ok = d_step < 1e-4 and d_mixed < 1e-4
print("PARITY PASS" if ok else "PARITY FAIL")
sys.exit(0 if ok else 1)
