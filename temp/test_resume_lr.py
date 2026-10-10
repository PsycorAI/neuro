"""test_resume_lr.py -- 2026-10-10 (Neuro 54). End-to-end CPU test of the RESUME LR fix in train_gpu.main.
Stage 1 trains 30 tiny steps (final ckpt at the end of its cosine: lr = ~0.10 x base). Stage 2 raises max_tokens to 60
steps and RESUMES. Its final ckpt must hold lr = lr_scale(59 of 60) x base = ~0.10 x base in every param group.
With the bug (base read after the optimizer state load) it would be ~0.10 x 0.10 = ~0.01 x base."""
import sys, os, shutil, yaml, glob
sys.path.insert(0, "/home/glenn/projects/neuro/src")
sys.argv = [sys.argv[0]]
src = open("/home/glenn/projects/neuro/temp/parity_r16_flags.py").read().split("from train_gpu import build_model")[0]
exec(src)                                   # torch reference for the LIF Triton kernels (CPU)
import torch
import train_gpu

c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/r21i_x1early_150m.yaml"))
c.update(d=64, n_neurons=128, d_mem=64, attention_heads=4, n_layers=4, attention_layer_positions=[1, 2, 3],
         attn_ffn_hidden=256, block_size=128, batch_size=2, eval_batch=2, eval_iters=3, eval_every=1000,
         ckpt_every=1000, log_every=10, fused_ce=False, amp=False, compile=False, gdn2_chunk_size=16, engram_v2_K=256,
         engram_v2_rank=8, mqar_batch=4, mqar_N_max=8, mqar_gap_max=40, mqar_pad_to=17, mqar_gap_pad_to=64,
         attn_ffn_ramp_start=0.1, attn_ffn_ramp_end=0.5, attn_chunk=64, run_name="smoke_resume_lr")
p = "/home/glenn/projects/neuro/temp/smoke_resume_lr.yaml"
ck = "/home/glenn/projects/neuro/models/checkpoints/smoke_resume_lr"
shutil.rmtree(ck, ignore_errors=True)
try:
    for steps in (30, 60):
        c["max_tokens"] = 2 * 128 * steps
        yaml.safe_dump({k: v for k, v in c.items() if v is not None}, open(p, "w"))
        sys.argv = ["train_gpu.py", "--config", p, "--device", "cpu"]
        train_gpu.main()
    last = sorted(glob.glob(ck + "/step_*.pt"))[-1]
    s = torch.load(last, map_location="cpu", weights_only=False)
    fresh = train_gpu._build_optimizers(train_gpu.build_model(dict(c)), c, "cpu")
    ratios = [g["lr"] / b["lr"] for st, o in zip(s["opt_states"], fresh) for g, b in zip(st["param_groups"], o.param_groups)]
    print(f"{os.path.basename(last)}: step {s['step']}, lr / base per group = {[round(r, 4) for r in ratios]}")
    assert s["step"] == 60 and all(0.09 < r < 0.12 for r in ratios), "resume changed the LR schedule"
    print("[OK] resume keeps the LR schedule (expected ~0.10 x base; the old bug gave ~0.01)")
finally:
    shutil.rmtree(ck, ignore_errors=True); os.path.exists(p) and os.remove(p)
