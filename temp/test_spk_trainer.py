"""test_spk_trainer.py -- 2026-09-30: trainer-loop checks for temp/stage_spklog/train_gpu.py (log fix + sparsity ramp).
 A. parity: control (signed stack), live x2 (atomics noise) vs staged -> per-step losses (ramp flag absent)
 B. smoke: staged with sparsity_ramp_start_frac 0.5 / sparsity_lambda_final 4.0 -> finite losses
Tiny runs (B=8, 20 steps) on GPU 1 beside the queue; checkpoints deleted at the end."""
import os, re, subprocess, shutil, yaml
ROOT = "/home/glenn/projects/neuro"; PY = "/home/glenn/projects/bdh/venv/bin/python"; STAGE = f"{ROOT}/temp/stage_spklog"
base = yaml.safe_load(open(f"{ROOT}/configs/screen_r8_signed_canon_s0_30m.yaml"))
def run(name, stage, **kw):
    shutil.rmtree(f"{ROOT}/models/checkpoints/{name}", ignore_errors=True)
    c = dict(base, batch_size=8, max_tokens=20 * 8 * 512, log_every=1, eval_every=10, eval_iters=3,
             ckpt_every=10 ** 9, keep_last=1, run_name=name, **kw)
    p = f"/tmp/{name}.yaml"; yaml.safe_dump(c, open(p, "w"))
    if stage:
        shim = (f"import sys; sys.path.insert(0, '{ROOT}/src'); sys.path.insert(0, '{STAGE}'); sys.argv = ['train_gpu.py', "
                f"'--config', '{p}', '--device', 'cuda']; import runpy; runpy.run_path('{STAGE}/train_gpu.py', run_name='__main__')")
        cmd = [PY, "-c", shim]
    else:
        cmd = [PY, f"{ROOT}/src/train_gpu.py", "--config", p, "--device", "cuda"]
    r = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="1"), capture_output=True, text=True)
    out = r.stdout + r.stderr[-1500:]
    return out, [float(x) for x in re.findall(r"^step \d+ \| .*? loss ([\d.]+)", out, re.M)]
res = {t: run(f"test_spk_{t}", t == "stage") for t in ("liveA", "liveB", "stage")}
L = {t: v[1] for t, v in res.items()}
print("A. n", {t: len(v) for t, v in L.items()}, "| step1", [round(v[0], 4) for v in L.values()])
print(f"   max|live-live| {max(abs(a - b) for a, b in zip(L['liveA'], L['liveB'])):.4f} | max|live-stage| "
      f"{max(abs(a - b) for a, b in zip(L['liveA'], L['stage'])):.4f}")
print("   spike log live :", re.findall(r"\[spike\].*", res["liveA"][0])[:1])
print("   spike log stage:", re.findall(r"\[spike\].*", res["stage"][0])[:1])
out, l = run("test_spk_ramp", True, sparsity_ramp_start_frac=0.5, sparsity_lambda_final=4.0)
print(f"B. ramp smoke: steps {len(l)}, finite {all(x == x for x in l)}, first/last {l[:1]} / {l[-1:]}")
if len(l) != 20: print(out[-2000:])
for d in os.listdir(f"{ROOT}/models/checkpoints"):
    if d.startswith("test_spk_"): shutil.rmtree(f"{ROOT}/models/checkpoints/{d}")
