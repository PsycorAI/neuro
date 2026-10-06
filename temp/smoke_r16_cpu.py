"""smoke_r16_cpu.py -- 2026-10-02. Runs the REAL train_gpu.main on CPU for ~40 tiny steps with every r16 flag on
(C0 shape + late gap dose; then X2 DSN-LIF; then T4 transformer) so a crash shows up here, not at night on the 5080.
LIF Triton kernels are swapped for the torch reference (as parity_r16_flags.py). Deletes its checkpoints."""
import sys, os, shutil, yaml
sys.path.insert(0, "/home/glenn/projects/neuro/src")
sys.argv = [sys.argv[0]]
import importlib.util
spec = importlib.util.spec_from_file_location("p16", "/home/glenn/projects/neuro/temp/parity_r16_flags.py")
import lif_exact as LE
src = open("/home/glenn/projects/neuro/temp/parity_r16_flags.py").read().split("from train_gpu import build_model")[0]
exec(src)                                   # defines lif_ml_ref and patches LE (stops before the tests)
import train_gpu

R14B = "/home/glenn/projects/neuro/configs/screen_r14b_3Lattn2_canon_ramp10_s0_30m.yaml"
base = yaml.safe_load(open(R14B))
base.update(d=64, n_neurons=128, d_mem=64, n_layers=4, attention_layer_positions=[1, 2, 3], attention_heads=4,
            block_size=64, batch_size=4, eval_batch=2, eval_iters=3, max_tokens=4 * 64 * 40, eval_every=20,
            ckpt_every=1000, log_every=10, fused_ce=False, amp=False, gdn2_chunk_size=16, engram_v2_K=256,
            engram_v2_rank=8)
r16 = dict(engram_v2_hash="hi", attn_out_gate=True, attn_canon=4, spk_out_norm=False, lif_levels=1,
           attn_topk=4, attn_ffn_spikeglu_k=16, attn_ffn_hidden=128, attn_ffn_spikeglu_ramp_start=0.3,
           attn_ffn_spikeglu_ramp_end=0.8, mqar_gap_max=40, mqar_gap_ramp_frac=0.1, mqar_batch=4,
           mqar_gap_start_frac=0.5, mqar_gap_weight=0.15, mqar_N_max=8)
CASES = {
    "smoke_r16_c0": dict(base, **r16),
    "smoke_r16_x2": dict(base, **dict(r16, attn_ffn_spikeglu_lif=True, attn_ffn_spikeglu_ramp_start=None)),
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
print("[OK] r16 CPU smoke passed")
