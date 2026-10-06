"""reeval_leakfree.py -- 2026-09-27: re-evaluate every checkpoint with the LEAK-FREE scan (fixed
src/lif_parallel.py) vs the old leaky bf16 scan (HEAD~2 version), using train_gpu.evaluate (same
EVAL_SEED windows as training-time evals). Prints one row per run."""
import sys, glob, os, subprocess, importlib.util, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import train_gpu
from train_gpu import build_model, evaluate, StreamText, BIN
import lif_parallel as LP
N = "/home/glenn/projects/neuro"
src = subprocess.run(["git", "-C", N, "show", "b11eb80:src/lif_parallel.py"], capture_output=True, text=True, check=True).stdout
open("/tmp/lif_leaky.py", "w").write(src)
sp = importlib.util.spec_from_file_location("lif_leaky", "/tmp/lif_leaky.py"); OLD = importlib.util.module_from_spec(sp); sp.loader.exec_module(OLD)
FIXED = LP._scan_chunked
torch.set_float32_matmul_precision("high")
runs = sys.argv[1:] or sorted(os.path.basename(d) for d in glob.glob(f"{N}/models/checkpoints/*")
                              if "lifx" not in d and "orpo" not in d)
print(f"{'run':34s} {'step':>7s} {'Mtok':>7s} | {'leaky ppl':>9s} | {'HONEST ppl':>10s} | {'honest ablated':>14s} | spike", flush=True)
for r in runs:
    cks = sorted(glob.glob(f"{N}/models/checkpoints/{r}/step_*.pt"))
    if not cks: continue
    s = torch.load(cks[-1], map_location="cpu", weights_only=False)
    c = s.get("cfg")
    if c is None: print(f"{r:34s} (no cfg in ckpt)"); continue
    c = dict(c); c.setdefault("eval_batch", c.get("batch_size", 8)); c["eval_batch"] = min(c["eval_batch"], 8)
    m = build_model(c).to("cuda"); m.load_state_dict(s["model"], strict=False); del s
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256))
    res = {}
    for name, scan in (("leaky", OLD._scan_chunked), ("honest", FIXED)):
        LP._scan_chunked = scan
        res[name] = evaluate(m, data, c, "cuda")
    LP._scan_chunked = FIXED
    step = int(cks[-1].split("_")[-1][:-3]); tok = step * c["batch_size"] * c.get("grad_accum", 1) * c["block_size"] / 1e6
    ab = res["honest"][1]
    print(f"{r:34s} {step:7d} {tok:7.0f} | {res['leaky'][0]:9.2f} | {res['honest'][0]:10.2f} | "
          f"{(f'{ab:.2f}' if ab else '-'):>14s} | {res['honest'][2] if res['honest'][2] is not None else '-'}", flush=True)
    del m; torch.cuda.empty_cache()
