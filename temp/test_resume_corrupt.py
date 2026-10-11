"""test_resume_corrupt.py -- 2026-10-11 (Neuro 55). CPU test of the two resume guards added to train_gpu:
  (1) stale step_*.pt.tmp files (a save cut off by an interruption) are removed at start;
  (2) an unreadable newest checkpoint falls back to the one before it; no readable checkpoint at all still raises.
Stage 1 trains 20 tiny steps with checkpoints at 10 and 20. Then the step-20 file is truncated and a fake .tmp is
planted; stage 2 (30 steps) must resume from step 10, remove the .tmp and finish. Stage 3: both files truncated -> error."""
import sys, os, shutil, yaml, glob, io, contextlib
sys.path.insert(0, "/home/glenn/projects/neuro/src")
sys.argv = [sys.argv[0]]
src = open("/home/glenn/projects/neuro/temp/parity_r16_flags.py").read().split("from train_gpu import build_model")[0]
exec(src)                                   # torch reference for the LIF Triton kernels (CPU)
import torch
import train_gpu

c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/v4_smoke1b.yaml"))
c.update(d=64, n_neurons=128, d_mem=64, attention_heads=4, attn_ffn_hidden=256, block_size=128, batch_size=2,
         eval_batch=2, eval_iters=3, eval_every=1000, ckpt_every=10, keep_last=3, log_every=5, fused_ce=False, amp=False,
         compile=False, gdn2_chunk_size=16, engram_v2_K=256, engram_v2_rank=8, mqar_batch=4, mqar_aug_batch=4,
         mqar_xchunk_batch=2, mqar_N_max=8, mqar_gap_max=40, mqar_pad_to=17, mqar_gap_pad_to=64, attn_chunk=64,
         run_name="smoke_resume_corrupt")
p = "/home/glenn/projects/neuro/temp/smoke_resume_corrupt.yaml"
ck = "/home/glenn/projects/neuro/models/checkpoints/smoke_resume_corrupt"
shutil.rmtree(ck, ignore_errors=True)


def run(steps):
    c["max_tokens"] = c["batch_size"] * c["block_size"] * int(c["grad_accum"]) * steps
    yaml.safe_dump({k: v for k, v in c.items() if v is not None}, open(p, "w"))
    buf = io.StringIO(); sys.argv = ["train_gpu.py", "--config", p, "--device", "cpu"]
    with contextlib.redirect_stdout(buf):
        train_gpu.main()
    return buf.getvalue()


def trunc(f):
    open(f, "r+b").truncate(os.path.getsize(f) // 3)


ok = True
try:
    run(20)
    files = sorted(glob.glob(ck + "/step_*.pt")); print("after stage 1:", [os.path.basename(f) for f in files])
    trunc(ck + "/step_0000020.pt"); open(ck + "/step_0000030.pt.tmp", "wb").write(b"x" * 1000)
    out = run(30)
    res = [l for l in out.splitlines() if l.startswith("resumed") or "[warn] cannot read" in l]
    print("\n".join(l[:150] for l in res))
    good = any("step_0000010.pt @ step 10" in l for l in res) and not glob.glob(ck + "/*.tmp") and os.path.exists(ck + "/step_0000030.pt")
    ok &= good; print("fallback to step 10, .tmp removed, finished at step 30:", "OK" if good else "FAIL")
    for f in glob.glob(ck + "/step_*.pt"):
        trunc(f)
    try:
        run(40); ok = False; print("all checkpoints unreadable: NO error raised -> FAIL")
    except RuntimeError as e:
        print("all checkpoints unreadable ->", str(e)[:70], ": OK")
finally:
    shutil.rmtree(ck, ignore_errors=True); os.path.exists(p) and os.remove(p)
print("[OK] resume guards work" if ok else "[FAIL]")
sys.exit(0 if ok else 1)
