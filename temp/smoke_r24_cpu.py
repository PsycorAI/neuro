"""smoke_r24_cpu.py -- 2026-10-04. Runs the REAL train_gpu.main on CPU for a few dozen tiny steps with the r24 flags on
the current hybrid stack (relu2 + homeo + top-k attention + gap MQAR): (a) stochastic attention window,
(b) stateful segments + reset_prob. Catches crashes before the 5090 queue reaches them. Deletes its checkpoints."""
import sys, os, shutil, yaml
sys.path.insert(0, "/home/glenn/projects/neuro/src")
sys.argv = [sys.argv[0]]
src = open("/home/glenn/projects/neuro/temp/parity_r16_flags.py").read().split("from train_gpu import build_model")[0]
exec(src)                                   # torch reference for the LIF Triton kernels (CPU)
import train_gpu

base = yaml.safe_load(open("/home/glenn/projects/neuro/configs/r21i_x1early_150m.yaml"))
base.update(d=64, n_neurons=128, d_mem=64, attention_heads=4, n_layers=4, attention_layer_positions=[1, 2, 3],
            attn_ffn_hidden=256, block_size=64, batch_size=4, eval_batch=2, eval_iters=3, max_tokens=4 * 64 * 60,
            eval_every=20, ckpt_every=1000, log_every=10, fused_ce=False, amp=False, compile=False,
            gdn2_chunk_size=16, engram_v2_K=256, engram_v2_rank=8, mqar_batch=4, mqar_N_max=8, mqar_gap_max=40,
            mqar_pad_to=17, mqar_gap_pad_to=64, attn_ffn_ramp_start=0.1, attn_ffn_ramp_end=0.5)
CASES = {
    "smoke_r24_swax": dict(base, attn_window_short=8, attn_window_p=0.5, attn_window_long_frac=0.1),
    "smoke_r24_stateful": dict(base, stateful_stream=True, reset_prob=0.1),
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
print("[OK] r24 CPU smoke passed")
