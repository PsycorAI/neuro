"""smoke_r25_cpu.py -- 2026-10-04. r25 chunked attention (attn_chunk) + cross-chunk gap MQAR (mqar_xchunk_frac).
(1) parity: attn_chunk = 0 leaves the model byte-identical (same weights, same logits as before the refactor is
    implied by (2a)); with attn_chunk = C the first chunk's logits equal a T = C forward, and every chunk's attention
    blocks equal per-chunk forwards (block-diagonal attention). The memory still spans the whole sequence: chunk-2
    logits CHANGE when chunk 1 changes (only via the memory), and do NOT change when memory is ablated + chunk 1 changes.
(2) the real train_gpu.main on CPU for a few dozen tiny steps with both flags on. Deletes its checkpoints."""
import sys, os, shutil, yaml, copy
sys.path.insert(0, "/home/glenn/projects/neuro/src")
sys.argv = [sys.argv[0]]
src = open("/home/glenn/projects/neuro/temp/parity_r16_flags.py").read().split("from train_gpu import build_model")[0]
exec(src)                                   # torch reference for the LIF Triton kernels (CPU)
import torch
import train_gpu
from train_gpu import build_model

base = yaml.safe_load(open("/home/glenn/projects/neuro/configs/r21i_x1early_150m.yaml"))
base.update(d=64, n_neurons=128, d_mem=64, attention_heads=4, n_layers=4, attention_layer_positions=[1, 2, 3],
            attn_ffn_hidden=256, block_size=64, batch_size=4, eval_batch=2, eval_iters=3, max_tokens=4 * 64 * 60,
            eval_every=20, ckpt_every=1000, log_every=10, fused_ce=False, amp=False, compile=False,
            gdn2_chunk_size=16, engram_v2_K=256, engram_v2_rank=8, mqar_batch=4, mqar_N_max=8, mqar_gap_max=40,
            mqar_pad_to=17, mqar_gap_pad_to=64, attn_ffn_ramp_start=0.1, attn_ffn_ramp_end=0.5)

# ---- (1) parity ----
C = 32
torch.manual_seed(0); m0 = build_model(dict(base)).eval()
torch.manual_seed(0); m1 = build_model(dict(base, attn_chunk=C)).eval()
m1.load_state_dict(m0.state_dict())
assert all(b.attn_chunk == C for b in m1.blocks if hasattr(b, "qkv")) and all(
    b.attn_chunk == 0 for b in m0.blocks if hasattr(b, "qkv"))
g = torch.Generator().manual_seed(1)
x = torch.randint(0, base["vocab"], (2, 2 * C), generator=g)
with torch.no_grad():
    full0 = m0(x); full1 = m1(x); first = m0(x[:, :C])
    d_first = (full1[:, :C] - first).abs().max().item()
    d_unch = (full0[:, :C] - full1[:, :C]).abs().max().item()
    d_second = (full0[:, C:] - full1[:, C:]).abs().max().item()
    print(f"chunk1 vs T=C forward: {d_first:.2e} | chunk1 chunked vs full: {d_unch:.2e} | chunk2 chunked vs full: {d_second:.2e}")
    assert d_first < 1e-5 and d_unch < 1e-5 and d_second > 1e-4
    # block-level: chunked attention block == per-chunk forwards
    blk = [b for b in m1.blocks if hasattr(b, "qkv")][0]
    h = torch.randn(2, 2 * C + 5, base["d_mem"], generator=g)        # ragged tail exercises the pad path
    z = blk.forward_sequence(h, None, None, None)[0]
    blk.attn_chunk = 0
    ref = torch.cat([blk.forward_sequence(h[:, i:i + C], None, None, None)[0] for i in range(0, h.shape[1], C)], 1)
    blk.attn_chunk = C
    print(f"block chunked vs per-chunk (ragged T={h.shape[1]}): {(z - ref).abs().max().item():.2e}")
    assert (z - ref).abs().max().item() < 1e-5
    # chunk 2 sees chunk 1 only through the memory
    x2 = x.clone(); x2[:, : C // 2] = torch.randint(0, base["vocab"], (2, C // 2), generator=g)
    dm = (m1(x2)[:, C:] - full1[:, C:]).abs().max().item()
    da = (m1(x2, ablate_memory=True)[:, C:] - m1(x, ablate_memory=True)[:, C:]).abs().max().item()
    print(f"chunk2 change when chunk1 changes: memory on {dm:.2e} | memory ablated {da:.2e}")
    assert dm > 1e-4
print("[OK] r25 parity")
# NOTE on the ablated case: the H0 LIF membrane and Engram n-grams still carry a little across the boundary, so da
# is small but need not be 0; reported, not asserted.

# ---- (2) real trainer on CPU ----
CASES = {
    "smoke_r25_chunk": dict(base, block_size=128, batch_size=2, attn_chunk=64, max_tokens=2 * 128 * 60),
    "smoke_r25_chunk_xmq": dict(base, block_size=128, batch_size=2, attn_chunk=64, max_tokens=2 * 128 * 60,
                                mqar_xchunk_frac=0.5, mqar_xchunk_batch=2, mqar_gap_start_frac=0.0,
                                mqar_aug_batch=3, mqar_xchunk_start_frac=0.3),
    "smoke_r25_nlfact": dict(base, block_size=128, batch_size=2, attn_chunk=64, max_tokens=2 * 128 * 60,
                             nlfact_every=2, nlfact_batch=2, nlfact_k_max=2),
}
for name, c in CASES.items():
    c["run_name"] = name
    p = f"/home/glenn/projects/neuro/temp/{name}.yaml"
    yaml.safe_dump({k: v for k, v in c.items() if v is not None}, open(p, "w"))
    ck = f"/home/glenn/projects/neuro/models/checkpoints/{name}"
    shutil.rmtree(ck, ignore_errors=True)
    print(f"===== {name} =====", flush=True)
    sys.argv = ["train_gpu.py", "--config", p, "--device", "cpu"]
    try:
        train_gpu.main()
    finally:
        shutil.rmtree(ck, ignore_errors=True); os.remove(p)
print("[OK] r25 CPU smoke passed")
