"""mk_kd_slice.py -- 2026-10-09 (Neuro 54). Raw (uint32 Llama-3 ids) slice of the v4 corpus for the cloud CONVERSION
phase: the Llama KD teacher needs raw ids, which the uint16 cloud corpus does not have. Slice = the first N tokens +
the exact val tail (data_val_tokens), so val windows match every other v4 eval. Read WITHOUT data_prelut.
  python temp/mk_kd_slice.py [N_TOKENS=2e9]   -> G:\\neuro_corpus_v4\\train_kd_slice_u32.bin (+ .sha256)"""
import sys, hashlib, numpy as np
SRC = "/home/glenn/projects/bdh/data/tokenized_v4/train.bin"
DST = "/mnt/g/neuro_corpus_v4/train_kd_slice_u32.bin"
N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 2_000_000_000
VAL = 14_527_194
src = np.memmap(SRC, dtype=np.uint32, mode="r")
assert N + VAL < len(src)
h = hashlib.sha256(); CH = 64_000_000
with open(DST, "wb") as f:
    for lo, hi in ((0, N), (len(src) - VAL, len(src))):
        for s in range(lo, hi, CH):
            b = np.asarray(src[s:min(s + CH, hi)]).tobytes(); f.write(b); h.update(b)
open(DST + ".sha256", "w").write(f"{h.hexdigest()}  train_kd_slice_u32.bin\n")
print(f"[OK] {DST}: {N + VAL:,} tokens ({(N + VAL) * 4 / 2**30:.2f} GiB), val tail {VAL:,}")
