"""test_r8_trainer.py -- 2026-09-29 trainer-loop tests for the staged r8 flags (tiny runs, B=8, beside a live job).
 A. parity: control config, live src x2 (atomics-noise reference) vs staged src -> per-step losses
 B. smoke (staged): stateful_stream, attn_window_short 64, both -> run + finite losses + MQAR aux active
 C. stateful resume (staged): 12 steps, relaunch to 25 -> "resumed" + cursors skipped (no crash)
Configs go to /tmp; checkpoints to models/checkpoints/test_r8_* (deleted at the end)."""
import os, re, subprocess, shutil, yaml
ROOT = "/home/glenn/projects/neuro"; PY = "/home/glenn/projects/bdh/venv/bin/python"
base = yaml.safe_load(open(f"{ROOT}/configs/screen_gap_hyb_full_s0_30m.yaml"))
STEP_TOK = 8 * 512

def cfg(name, steps, **kw):
    c = dict(base, batch_size=8, max_tokens=steps * STEP_TOK, log_every=1, eval_every=10 ** 9, eval_iters=3,
             ckpt_every=10 ** 9, keep_last=1, run_name=name, **kw)
    p = f"/tmp/{name}.yaml"; yaml.safe_dump(c, open(p, "w")); return p

def run(name, steps, stage, fresh=True, **kw):
    if fresh: shutil.rmtree(f"{ROOT}/models/checkpoints/{name}", ignore_errors=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", PYTHONUNBUFFERED="1")
    # train_gpu is run as a script: its own dir (src/) is sys.path[0], so the stage must come first via a shim
    shim = f"import sys; sys.path.insert(0, '{ROOT}/src'); sys.path.insert(0, '{ROOT}/temp/stage_r8'); sys.argv = ['train_gpu.py', '--config', '{cfg(name, steps, **kw)}', '--device', 'cuda']; import runpy; runpy.run_path('{ROOT}/temp/stage_r8/train_gpu.py', run_name='__main__')" if stage else None
    cmd = [PY, "-c", shim] if stage else [PY, f"{ROOT}/src/train_gpu.py", "--config", f"/tmp/{name}.yaml", "--device", "cuda"]
    if not stage: cfg(name, steps, **kw)
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    out = p.stdout + "\n[stderr]\n" + p.stderr[-2000:]
    losses = [float(x) for x in re.findall(r"^step \d+ \| .*? loss ([\d.]+)", out, re.M)]
    return out, losses

res = {}
for tag, stage in (("liveA", False), ("liveB", False), ("stage", True)):
    out, l = run(f"test_r8_par_{tag}", 20, stage); res[tag] = l
    if not l: print(out[-3000:])
print("A. parity (20 steps): n", {k: len(v) for k, v in res.items()})
dAB = max(abs(a - b) for a, b in zip(res["liveA"], res["liveB"]))
dAS = max(abs(a - b) for a, b in zip(res["liveA"], res["stage"]))
print(f"   step1 losses {res['liveA'][0]:.4f} / {res['liveB'][0]:.4f} / {res['stage'][0]:.4f} | max|live-live| {dAB:.4f} "
      f"| max|live-stage| {dAS:.4f}  -> PASS if live-stage is at the live-live noise level")

for tag, kw in (("stateful", dict(stateful_stream=True)), ("window", dict(attn_window_short=64, attn_window_p=0.5)),
                ("both", dict(stateful_stream=True, attn_window_short=64, attn_window_p=0.5))):
    out, l = run(f"test_r8_smoke_{tag}", 20, True, **kw)
    flags = [s for s in ("STATEFUL streaming enabled", "stochastic attention window") if s in out]
    ok = len(l) == 20 and all(x == x and x < 20 for x in l)
    print(f"B. smoke {tag:8s}: {'OK' if ok else 'FAIL'} steps={len(l)} first/last loss {l[0] if l else None} / "
          f"{l[-1] if l else None} | banners {flags}")
    if not ok: print(out[-3000:])

out1, l1 = run("test_r8_resume", 12, True, stateful_stream=True)
out2, l2 = run("test_r8_resume", 25, True, fresh=False, stateful_stream=True)
m = re.search(r"resumed .*", out2)
print(f"C. resume: first leg {len(l1)} steps, second leg {len(l2)} steps | {m.group(0) if m else 'NO RESUME LINE'}")
if not m or len(l2) != 13: print(out2[-3000:])
for d in os.listdir(f"{ROOT}/models/checkpoints"):
    if d.startswith("test_r8_"): shutil.rmtree(f"{ROOT}/models/checkpoints/{d}")
print("cleaned test checkpoints")
