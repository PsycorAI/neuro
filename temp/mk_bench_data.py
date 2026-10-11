"""mk_bench_data.py -- 2026-10-11 (Neuro 55). Two SMALL data files so cloud/bench_rc3.sh can run on the instance before
(or without) the Drive corpus download and before the 7.5 GB KD slice is uploaded: throughput and memory do not depend
on which tokens are read. Each = the first N tokens + the exact val tail (data_val_tokens), same layout as the real files.
  data/bench_u16.bin  (uint16, LUT applied -> data_prelut: true)    stands in for train_lut32768_u16.bin
  data/bench_u32.bin  (uint32 raw Llama-3 ids -> data_prelut: false) stands in for train_kd_slice_u32.bin
cloud/push_from_home.sh sends both (~210 MB, ~20 min on the DSL line)."""
import numpy as np
T4 = "/home/glenn/projects/bdh/data/tokenized_v4/"
OUT = "/home/glenn/projects/neuro/data/"
N, VAL = 20_000_000, 14_527_194
for src, dt, dst in ((T4 + "train_lut32768_u16.bin", np.uint16, "bench_u16.bin"), (T4 + "train.bin", np.uint32, "bench_u32.bin")):
    m = np.memmap(src, dtype=dt, mode="r")
    with open(OUT + dst, "wb") as f:
        f.write(np.asarray(m[:N]).tobytes()); f.write(np.asarray(m[len(m) - VAL:]).tobytes())
    print(f"[OK] {OUT + dst}: {N + VAL:,} tokens, {(N + VAL) * np.dtype(dt).itemsize / 2**20:.0f} MiB")
