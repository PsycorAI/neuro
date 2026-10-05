"""smoke_r19_cpu.py -- 2026-10-03. (1) SparseAct unit test: the homeostatic threshold drives the firing rate to the
target ("layer" and "unit"), and with homeo off relu2 is relu(a)^2. (2) Runs the REAL train_gpu.main on CPU for a
few dozen tiny steps on the l3 (GELU) config with r19 flags (relu2; relu2 + layer/unit homeo, fast ramp) so a crash
shows up here and not overnight. (3) gelu default builds an nn.GELU (byte-identical path). Deletes its checkpoints."""
import sys, os, shutil, yaml
import torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
sys.argv = [sys.argv[0]]
import lif_exact as LE
src = open("/home/glenn/projects/neuro/temp/parity_r16_flags.py").read().split("from train_gpu import build_model")[0]
exec(src)                                   # torch reference for the LIF Triton kernels (CPU)
import train_gpu
from hybrid_attn import SparseAct, HybridAttentionBlock

# (1) unit test
torch.manual_seed(0)
a = torch.randn(64, 32, 256)
m = SparseAct(256, "relu2")
assert torch.equal(m(a), torch.relu(a) ** 2)
for homeo in ("layer", "unit"):
    m = SparseAct(256, "relu2", homeo); m.train(); m.rate_target = 0.1
    for _ in range(3000):
        m(torch.randn(64, 32, 256) * (1 + torch.arange(256) / 256)); m.homeo_step(0.01)
    r = (m(a * (1 + torch.arange(256) / 256)) > 0).float()     # same input distribution the threshold adapted to
    print(f"[unit] {homeo}: rate {r.mean():.3f} (target 0.1) | per-unit min/max {r.mean((0, 1)).min():.3f}/{r.mean((0, 1)).max():.3f}")
    assert abs(r.mean().item() - 0.1) < 0.02
# (3) default = nn.GELU
blk = HybridAttentionBlock(d_in=64, d_mem=64, n_neurons=32, n_heads=4, use_swiglu=False)
assert isinstance(blk.ffn[1], torch.nn.GELU)

# (2) real trainer
L3 = "/home/glenn/projects/neuro/configs/screen_r16_l3_gelu_s0_30m.yaml"
base = yaml.safe_load(open(L3))
base.update(d=64, n_neurons=128, d_mem=64, attention_heads=4, block_size=64, batch_size=4, eval_batch=2,
            eval_iters=3, max_tokens=4 * 64 * 60, eval_every=20, ckpt_every=1000, log_every=10, fused_ce=False,
            amp=False, gdn2_chunk_size=16, engram_v2_K=256, engram_v2_rank=8, mqar_batch=4, mqar_N_max=8)
CASES = {
    "smoke_r19_relu2": dict(base, attn_ffn_act="relu2"),
    "smoke_r19_layer": dict(base, attn_ffn_act="relu2", attn_ffn_homeo="layer", attn_ffn_rate_target=0.1,
                            attn_ffn_ramp_start=0.1, attn_ffn_ramp_end=0.5, attn_ffn_homeo_lr=0.05),
    "smoke_r19_unit": dict(base, attn_ffn_act="relu2", attn_ffn_homeo="unit", attn_ffn_rate_target=0.1,
                           attn_ffn_ramp_start=0.1, attn_ffn_ramp_end=0.5, attn_ffn_homeo_lr=0.05),
    # 2026-10-03: GA2 + mqar_once_per_step (MQAR on the first micro-step only, batch x2, weight x2)
    "smoke_r19_mqonce": dict(base, attn_ffn_act="relu2", attn_ffn_homeo="layer", attn_ffn_rate_target=0.1,
                             grad_accum=2, mqar_once_per_step=True),
    # 2026-10-04: fixed attention window (train + eval) + gentle gap MQAR beyond the window
    "smoke_r21_win": dict(base, attn_ffn_act="relu2", attn_ffn_homeo="layer", attn_ffn_rate_target=0.1,
                          attn_window_fixed=16, mqar_gap_max=40, mqar_gap_weight=0.15, mqar_gap_start_frac=0.3,
                          mqar_pad_to=17, mqar_gap_pad_to=64),
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
print("[OK] r19 CPU smoke passed")
