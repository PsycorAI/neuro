"""bench_loader2.py -- 2026-09-30: cheap, byte-identical fixes for the independent-windows loader (1227 ms/batch cold).
Same 32 random starts each way; compares plain order, sorted-offset order, and an 8-thread pool. CPU only."""
import time, numpy as np
from concurrent.futures import ThreadPoolExecutor
BIN = "/home/glenn/projects/bdh/data/tokenized/train.bin"
d = np.memmap(BIN, dtype=np.uint32, mode="r")
B, T, N = 32, 512, 20
rng = np.random.default_rng(int(time.time()))
pool = ThreadPoolExecutor(8)
def plain(st):  return np.stack([np.asarray(d[s:s + T + 1]) for s in st])
def srt(st):
    o = np.argsort(st); out = np.empty((B, T + 1), np.uint32)
    for i in o: out[i] = d[st[i]:st[i] + T + 1]
    return out
def thr(st):    return np.stack(list(pool.map(lambda s: np.asarray(d[s:s + T + 1]), st)))
for name, f in (("plain", plain), ("sorted", srt), ("8 threads", thr)):
    starts = [rng.integers(0, int(len(d) * 0.999) - T - 1, size=B) for _ in range(N)]   # fresh (cold) offsets per method
    t = time.time()
    for st in starts: f(st)
    print(f"{name:10s} {1000 * (time.time() - t) / N:7.1f} ms/batch", flush=True)
st = rng.integers(0, int(len(d) * 0.999) - T - 1, size=B)
assert np.array_equal(plain(st), srt(st)) and np.array_equal(plain(st), thr(st)); print("identical batches: OK")
