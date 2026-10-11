"""profile_train.py -- 2026-09-27: torch.profiler over the REAL train_gpu loop, non-invasively.

Wraps train_gpu.main() (no edits to shared modules -- the 5090 post-training imports them):
  * a global optimizer-step post-hook drives prof.step() once per training step
  * lif_parallel is wrapped in record_function("LIF_PARALLEL") and counts its scans
    (scans per call = fixed-point iterations + 1 final scan with grad)

  CUDA_VISIBLE_DEVICES=1 python temp/profile_train.py --config configs/flagship_v4_363m.draft.yaml \
      --tag v4d --batch 8 --set key=val ...
"""
import os, sys, argparse, yaml, time, collections
ROOT = "/home/glenn/projects/neuro"
sys.path.insert(0, os.path.join(ROOT, "src"))
import torch
import torch.profiler as tp

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--grad_accum", type=int, default=1)
ap.add_argument("--wait", type=int, default=12)    # compile/autotune warm-up steps (not profiled)
ap.add_argument("--active", type=int, default=6)
ap.add_argument("--noprof", action="store_true")   # timing only (no profiler overhead)
ap.add_argument("--set", nargs="*", default=[])    # extra key=val (yaml-parsed) overrides
ap.add_argument("--lif_nosync", action="store_true")
ap.add_argument("--amb", type=float, default=None)   # torch._functorch.config.activation_memory_budget (compile-time SAC)  # bit-identical FPT without the torch.equal early exit
a = ap.parse_args()

c = yaml.safe_load(open(a.config))
steps = a.wait + 2 + a.active + 3
c.update(run_name=f"screen_prof_{a.tag}", batch_size=a.batch, grad_accum=a.grad_accum,
         eval_every=10**9, ckpt_every=10**9, log_every=5,
         max_tokens=a.batch * a.grad_accum * c["block_size"] * steps)
for kv in a.set:
    k, v = kv.split("=", 1)
    c[k] = yaml.safe_load(v)
cfg = f"{ROOT}/temp/prof_{a.tag}.yaml"
yaml.safe_dump(c, open(cfg, "w"))

if a.amb is not None:
    import torch._functorch.config as _fc
    _fc.activation_memory_budget = a.amb
    print(f"activation_memory_budget = {a.amb}")
import lif_parallel as LP
stats = collections.Counter()
_orig_lif, _orig_scan = LP.lif_parallel, LP._scan_chunked
def _scan(*args, **kw):
    stats["scans"] += 1
    return _orig_scan(*args, **kw)
LP._scan_chunked = _scan
@torch.compiler.disable
def _lif(*args, **kw):
    stats["lif_calls"] += 1
    with tp.record_function("LIF_PARALLEL"):
        return _orig_lif(*args, **kw)
LP.lif_parallel = _lif
if a.lif_nosync:
    import math
    import torch.nn.functional as F
    @torch.compiler.disable
    def _orig_lif(cur, beta, vth=1.0, K=3, alpha=2.0):
        with torch.no_grad():
            s = torch.zeros_like(cur)
            for _ in range(K):
                s = (LP._scan_chunked(beta, cur - beta * vth * F.pad(s[:, :-1, :], (0, 0, 1, 0))) > vth).to(cur.dtype)
        pre_mem = LP._scan_chunked(beta, cur - beta * vth * F.pad(s[:, :-1, :], (0, 0, 1, 0)))
        over = pre_mem - vth
        spk_hard = (over > 0).to(cur.dtype)
        g = (1.0 / (math.pi * alpha)) * torch.atan(math.pi * alpha * over)
        return spk_hard.detach() + g - g.detach(), pre_mem[:, -1, :] - spk_hard[:, -1, :] * vth

prof = None
if not a.noprof:
    prof = tp.profile(activities=[tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA],
                      schedule=tp.schedule(wait=a.wait, warmup=2, active=a.active, repeat=1))
    prof.__enter__()
first_opt = {}
step_t = []
def _hook(opt, args, kw):
    if not first_opt:
        first_opt["id"] = id(opt)
    if id(opt) == first_opt["id"]:
        torch.cuda.synchronize(); step_t.append(time.time())
        if prof is not None:
            prof.step()
from torch.optim.optimizer import register_optimizer_step_post_hook
register_optimizer_step_post_hook(_hook)

import train_gpu
sys.argv = ["train_gpu.py", "--config", cfg, "--device", "cuda"]
train_gpu.main()

if prof is not None:
    prof.__exit__(None, None, None)
    ka = prof.key_averages()
    print(ka.table(sort_by="self_cuda_time_total", row_limit=45, max_name_column_width=70))
    tot = sum(e.self_device_time_total for e in ka)
    lif = [e for e in ka if e.key == "LIF_PARALLEL"]
    if lif:
        print(f"LIF_PARALLEL device time (incl. children) = {lif[0].device_time_total/1e3:.1f} ms "
              f"of {tot/1e3:.1f} ms total self device time ({100*lif[0].device_time_total/max(tot,1):.1f}%)")
    prof.export_chrome_trace(f"{ROOT}/logs/prof_{a.tag}.trace.json")
dts = [t1 - t0 for t0, t1 in zip(step_t, step_t[1:])][a.wait:]
tok = a.batch * a.grad_accum * c["block_size"]
if dts:
    dts.sort(); med = dts[len(dts) // 2]
    print(f"[prof {a.tag}] median step {med*1e3:.0f} ms = {tok/med:,.0f} tok/s (B{a.batch}xGA{a.grad_accum}, "
          f"{len(dts)} steps{', PROFILER ON' if prof else ''})")
print(f"[prof {a.tag}] lif_calls={stats['lif_calls']} scans={stats['scans']} "
      f"-> mean fixed-point iters/call = {stats['scans']/max(stats['lif_calls'],1) - 1:.2f}")
# 2026-10-11 (Neuro 55): true peak, for batch sizing on the H200 (the trainer's "vram a/b" = allocated NOW / reserved)
if torch.cuda.is_available():
    print(f"[prof {a.tag}] peak allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GB | peak reserved "
          f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GB")
