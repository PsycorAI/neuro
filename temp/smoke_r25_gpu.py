"""smoke_r25_gpu.py -- 2026-10-04. Tiny r25 run on the GPU with the REAL compiled path (compile, bf16, fused CE, static
MQAR compile, eager cumsum) and the xchunk MQAR from step 0, so a compile/shape bug can't kill a queued r25 run late
(r25a's xchunk dose starts at 70%). Small width -> ~1-2 GB, a few minutes. Deletes its checkpoints."""
import sys, os, shutil, yaml
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import train_gpu

for src, name in (("r25d_chunk_x1e_read_150m", "smoke_r25g_rd"),):
    c = yaml.safe_load(open(f"/home/glenn/projects/neuro/configs/{src}.yaml"))
    c.update(d=128, n_neurons=256, d_mem=128, attention_heads=4, n_layers=4, attention_layer_positions=[1, 2, 3], mem_read_layers=[2],
             attn_ffn_hidden=512, max_tokens=c["block_size"] * c["batch_size"] * 120, eval_every=40, eval_iters=3,
             ckpt_every=1000, log_every=20, engram_v2_K=1024, engram_v2_rank=16, mqar_gap_start_frac=0.0,
             attn_ffn_ramp_start=0.1, attn_ffn_ramp_end=0.5, run_name=name)
    p = f"/home/glenn/projects/neuro/temp/{name}.yaml"
    yaml.safe_dump(c, open(p, "w"))
    ck = f"/home/glenn/projects/neuro/models/checkpoints/{name}"
    shutil.rmtree(ck, ignore_errors=True)
    print(f"===== {name} (from {src}) =====", flush=True)
    sys.argv = ["train_gpu.py", "--config", p, "--device", "cuda"]
    try:
        train_gpu.main()
    finally:
        shutil.rmtree(ck, ignore_errors=True); os.remove(p)
print("[OK] r25 GPU smoke passed")
