"""bench_loader3.py -- 2026-09-30: does reading MANY pre-drawn windows in file order amortize the seek cost?
(If yes: a background 'window bank' that pre-draws K steps of starts, reads them sorted, serves them in original order
= byte-identical batches at a fraction of the IO time.) Reports ms per 32-window batch-equivalent. CPU only."""
import time, numpy as np
BIN = "/home/glenn/projects/bdh/data/tokenized/train.bin"
d = np.memmap(BIN, dtype=np.uint32, mode="r")
T = 512
rng = np.random.default_rng(int(time.time()))
hi = int(len(d) * 0.999) - T - 1
for n in (512, 4096, 16384):
    st = np.sort(rng.integers(0, hi, size=n))
    t = time.time()
    out = np.empty((n, T + 1), np.uint32)
    for i, s in enumerate(st): out[i] = d[s:s + T + 1]
    dt = time.time() - t
    print(f"{n:6d} windows sorted: {dt:6.1f} s total = {1000 * dt / (n / 32):7.1f} ms per 32-window batch "
          f"(avg gap {len(d) * 4 / n / 1e6:.1f} MB)", flush=True)
