"""test_resume_loss.py -- 2026-10-10 (Neuro 55). CPU check of the 1B-smoke anomaly: logged train loss at the same
steps was ~12.2 in the fresh process and ~7.0 after the resume at step 50 (logs/v4_smoke1b.log).
Tiny copy of configs/v4_smoke1b.yaml: run A = 20 steps fresh; run B = resume from A's step-10 ckpt to step 20.
Prints the logged loss per step for both; they should match closely at steps 11-20."""
import sys, os, shutil, yaml, glob, io, re, contextlib
sys.path.insert(0, "/home/glenn/projects/neuro/src")
sys.argv = [sys.argv[0]]
src = open("/home/glenn/projects/neuro/temp/parity_r16_flags.py").read().split("from train_gpu import build_model")[0]
exec(src)                                   # torch reference for the LIF Triton kernels (CPU)
import torch
import train_gpu

c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/v4_smoke1b.yaml"))
L = int(c["n_layers"]); print("base layout: n_layers", L, "attn positions", c.get("attention_layer_positions"))
c.update(d=64, n_neurons=128, d_mem=64, attention_heads=4, attn_ffn_hidden=256, block_size=128, batch_size=2,
         eval_batch=2, eval_iters=3, eval_every=1000, ckpt_every=10, log_every=1, fused_ce=False, amp=False,
         compile=False, gdn2_chunk_size=16, engram_v2_K=256, engram_v2_rank=8, mqar_batch=4, mqar_aug_batch=4,
         mqar_xchunk_batch=2, mqar_N_max=8, mqar_gap_max=40, mqar_pad_to=17, mqar_gap_pad_to=64, attn_chunk=64,
         run_name="smoke_resume_loss")
p = "/home/glenn/projects/neuro/temp/smoke_resume_loss.yaml"
ck = "/home/glenn/projects/neuro/models/checkpoints/smoke_resume_loss"
shutil.rmtree(ck, ignore_errors=True)
c["max_tokens"] = c["batch_size"] * c["block_size"] * int(c.get("grad_accum", 1)) * 20
yaml.safe_dump({k: v for k, v in c.items() if v is not None}, open(p, "w"))


REC = []
_orig = train_gpu.mqar_aug_batch
def _rec(*a, **k):
    r = _orig(*a, **k); REC.append((a[1], int(r[0].sum()))); return r
train_gpu.mqar_aug_batch = _rec


def run():
    REC.clear()
    buf = io.StringIO()
    sys.argv = ["train_gpu.py", "--config", p, "--device", "cpu"]
    with contextlib.redirect_stdout(buf):
        train_gpu.main()
    out = buf.getvalue()
    return {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"^step (\d+) \|.*?loss ([0-9.]+)", out, re.M)}, out


try:
    a, out_a = run(); rec_a = list(REC)
    for f in sorted(glob.glob(ck + "/step_*.pt")):
        if not f.endswith("step_0000010.pt"):
            os.remove(f)
    assert glob.glob(ck + "/step_0000010.pt"), "no step-10 ckpt:\n" + out_a[-2000:]
    b, out_b = run(); rec_b = list(REC)
    print("plain-MQAR batches (N, token sum): fresh first 6 ", rec_a[:6])
    print("                                   resumed first 6", rec_b[:6])
    print("resumed batches that repeat a fresh-run batch from steps 1-10:", sum(r in rec_a[:len(rec_a)] for r in rec_b), "of", len(rec_b))
    print([l for l in out_b.splitlines() if l.startswith("resumed")])
    print("step  fresh  resumed  diff")
    for s in sorted(b):
        print(f"{s:4d} {a.get(s, float('nan')):6.3f} {b[s]:7.3f} {b[s] - a.get(s, float('nan')):+6.3f}")
finally:
    shutil.rmtree(ck, ignore_errors=True); os.path.exists(p) and os.remove(p)
