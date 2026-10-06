"""bench_loader.py -- 2026-09-30: is the 30M screen data-bound? r9a (SequentialStream) ran 30-35k tok/s vs 14-15k for
random-window screens with the same model. Times one train batch (B=32, T=512) per sampler, CPU only, cold-ish cache."""
import sys, time, numpy as np
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from data_stream import StreamText, SequentialStream
BIN = "/home/glenn/projects/bdh/data/tokenized/train.bin"
B, T, N = 32, 512, 30
for name, mk in (("independent_windows (current screens)", lambda: StreamText(BIN, vocab=16384, independent_windows=True)),
                 ("region sampler (old default)", lambda: StreamText(BIN, vocab=16384, independent_windows=False))):
    d = mk(); np.random.seed(int(time.time()) % 100000)
    t = time.time()
    for _ in range(N): d.batch(B, T, "train", "cpu")
    print(f"{name:40s} {1000 * (time.time() - t) / N:7.1f} ms/batch")
s = SequentialStream(BIN, vocab=16384, batch_size=B, seed=int(time.time()) % 1000)
t = time.time()
for _ in range(N): s.next(T, "cpu")
print(f"{'SequentialStream (stateful)':40s} {1000 * (time.time() - t) / N:7.1f} ms/batch")
print(f"step time at 14.5k tok/s = {1000 * B * T / 14500:.0f} ms; at 32k tok/s = {1000 * B * T / 32000:.0f} ms")
