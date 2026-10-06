"""test_madv.py -- 2026-09-30: staged MADV_RANDOM loader (temp/stage_madv/data_stream.py) vs live src/data_stream.py:
byte-identical train/val batches for the same np.random seeds (both samplers), and train-batch timing. CPU only."""
import sys, time, importlib.util, numpy as np, torch
BIN = "/home/glenn/projects/bdh/data/tokenized/train.bin"
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
live = load("/home/glenn/projects/neuro/src/data_stream.py", "ds_live")
stage = load("/home/glenn/projects/neuro/temp/stage_madv/data_stream.py", "ds_stage")
ok = True
for indep in (True, False):
    a = live.StreamText(BIN, vocab=16384, independent_windows=indep)
    b = stage.StreamText(BIN, vocab=16384, independent_windows=indep)
    for split in ("train", "val"):
        for seed in range(5):
            np.random.seed(seed); xa, ya = a.batch(32, 512, split)
            np.random.seed(seed); xb, yb = b.batch(32, 512, split)
            ok &= torch.equal(xa, xb) and torch.equal(ya, yb)
    np.random.seed(123); _, _, ra = a.batch(8, 512, "train", return_raw=True)
    np.random.seed(123); _, _, rb = b.batch(8, 512, "train", return_raw=True)
    ok &= torch.equal(ra, rb)
print("byte-identical batches (indep on/off x train/val x 5 seeds + raw):", ok)
for name, mod in (("live", live), ("staged", stage)):
    d = mod.StreamText(BIN, vocab=16384, independent_windows=True); np.random.seed(int(time.time()) % 99991)
    t = time.time()
    for _ in range(10): d.batch(32, 512, "train")
    print(f"{name:7s} independent-windows train batch: {100 * (time.time() - t):7.1f} ms")
