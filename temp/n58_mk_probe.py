"""n58_mk_probe.py -- 2026-10-12 (Neuro 58). configs/probe_v4_xchunk.yaml = v4_cloud resumed AT HOME from step 46,500
with the cross-chunk recall training switched on NOW (in the cloud run it starts at 70%: xchunk batches sit inside the
gap branch, train_gpu.py:1180, so mqar_xchunk_start_frac 0.5 is gated by mqar_gap_start_frac 0.7).
Question: can the memory layer learn exact cross-chunk recall from a 0.1% spike rate?
Same tokens per optimizer step as the cloud (B1 x GA16 x 2048 = 32,768) -> same total_steps, same LR / lam / sparsity
schedule position; the aux batch runs every micro-step, so 16 x batch 1 = the cloud's batch 16 per step.
Differences: xchunk share 0.5 instead of 0.25 (twice the dose), full gap at once, aux batches of 1, frequent evals.
ARGS: [name] [key=value ...] extra overrides (for the spike-floor arm)."""
import sys, yaml
c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/v4_cloud.yaml"))
name = sys.argv[1] if len(sys.argv) > 1 and "=" not in sys.argv[1] else "probe_v4_xchunk"
c.update(run_name=name, batch_size=1, grad_accum=16, eval_batch=1, eval_iters=10,
         data_bin="/home/glenn/projects/bdh/data/tokenized_v4/train_lut32768_u16.bin",
         mqar_batch=1, mqar_xchunk_batch=1, mqar_aug_batch=4, mqar_aux_splits=1,
         mqar_gap_start_frac=0.2, mqar_gap_ramp_frac=0.0, mqar_xchunk_start_frac=0.0, mqar_xchunk_frac=0.5,
         eval_every=50, ckpt_every=50, keep_last=2, log_every=5)
for kv in sys.argv[1:]:
    if "=" in kv:
        k, v = kv.split("=", 1); c[k] = yaml.safe_load(v)
out = f"/home/glenn/projects/neuro/configs/{name}.yaml"
with open(out, "w") as f:
    f.write(f"# {name} -- AUTO-GEN temp/n58_mk_probe.py (2026-10-12): home probe, v4_cloud step 46,500 + cross-chunk recall now.\n")
    yaml.safe_dump(c, f, sort_keys=False)
print("wrote", out)
