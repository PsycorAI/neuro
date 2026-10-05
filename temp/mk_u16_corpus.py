"""mk_u16_corpus.py -- 2026-10-03 (Glenn OK): uint16 copy of tokenized_v4/train.bin with the vocab-32768 LUT applied,
for the Vast upload (46 GB instead of 93 GB). Read with `data_prelut: true` (src/data_stream.py).
Then checks parity: StreamText on both files gives identical train + val batches (both samplers)."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from data_stream import StreamText

SRC = "/home/glenn/projects/bdh/data/tokenized_v4/train.bin"
DST = "/home/glenn/projects/bdh/data/tokenized_v4/train_lut32768_u16.bin"
VOCAB = 32768
lut = np.load(f"/home/glenn/projects/neuro/data/vocab_map_{VOCAB}.npy")
assert lut.max() < VOCAB <= 65536 and lut.min() >= 0
src = np.memmap(SRC, dtype=np.uint32, mode="r")
n = len(src)
if not (os.path.exists(DST) and os.path.getsize(DST) == 2 * n):
    dst = np.memmap(DST + ".part", dtype=np.uint16, mode="w+", shape=(n,))
    CH = 256 * 1024 * 1024
    t0 = time.time()
    for i in range(0, n, CH):
        dst[i:i + CH] = lut[np.asarray(src[i:i + CH], dtype=np.int64)].astype(np.uint16)
        if (i // CH) % 8 == 0:
            print(f"[u16] {i / n:6.1%}  {time.time() - t0:6.0f}s", flush=True)
    dst.flush(); del dst
    os.rename(DST + ".part", DST)
print(f"[u16] written {DST} ({os.path.getsize(DST) / 1e9:.1f} GB, {n:,} tokens)")

# parity: same seeds -> identical batches from the uint32+LUT path and the prelut path
vt = 14527194
for iw in (False, True):
    a = StreamText(SRC, vocab=VOCAB, independent_windows=iw, val_tokens=vt)
    b = StreamText(DST, vocab=VOCAB, independent_windows=iw, val_tokens=vt, prelut=True)
    assert a.split == b.split
    for split in ("train", "val"):
        np.random.seed(123); xa, ya = a.batch(8, 512, split)
        np.random.seed(123); xb, yb = b.batch(8, 512, split)
        assert (xa == xb).all() and (ya == yb).all(), (iw, split)
# full-file spot check at 1000 random offsets
rng = np.random.default_rng(0); d = np.memmap(DST, dtype=np.uint16, mode="r")
for o in rng.integers(0, n - 4096, 1000):
    assert (lut[np.asarray(src[o:o + 4096], dtype=np.int64)] == d[o:o + 4096]).all()
json.dump({"format": "flat_token_stream", "dtype": "uint16", "lut": f"vocab_map_{VOCAB}.npy (applied)",
           "vocab": VOCAB, "total_tokens": int(n), "source": SRC, "data_val_tokens": vt,
           "read_with": "data_prelut: true", "built": time.strftime("%Y-%m-%d %H:%M")},
          open(DST.replace(".bin", ".json"), "w"), indent=1)
print("[u16] PARITY OK (both samplers, train+val, 1000 spot checks)")
