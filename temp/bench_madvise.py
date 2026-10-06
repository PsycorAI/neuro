"""bench_madvise.py -- 2026-09-30: does MADV_RANDOM on the train.bin memmap (read-ahead off for this mapping only) fix the
~37 ms cold random window read (device read_ahead_kb = 8192)? Per-read latency + a 32-window batch, fresh offsets. CPU only."""
import time, mmap, numpy as np
BIN = "/home/glenn/projects/bdh/data/tokenized/train.bin"
rng = np.random.default_rng(int(time.time()))
for label, adv in (("default read-ahead", False), ("MADV_RANDOM", True)):
    d = np.memmap(BIN, dtype=np.uint32, mode="r")
    if adv: d._mmap.madvise(mmap.MADV_RANDOM)
    st = rng.integers(0, len(d) - 600, size=64); lat = []
    for s in st:
        t0 = time.perf_counter(); np.asarray(d[s:s + 513]).sum(); lat.append(time.perf_counter() - t0)
    lat = np.array(lat) * 1000
    b = rng.integers(0, len(d) - 600, size=(10, 32)); t0 = time.perf_counter()
    for row in b: np.stack([np.asarray(d[s:s + 513]) for s in row])
    print(f"{label:20s} per-read median {np.median(lat):6.2f} ms p90 {np.percentile(lat, 90):6.2f} | 32-window batch "
          f"{1000 * (time.perf_counter() - t0) / 10:7.1f} ms", flush=True)
    del d
