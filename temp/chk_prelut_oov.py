"""chk_prelut_oov.py -- 2026-10-09 (Neuro 54). How many tokens of the uint16 (prelut) cloud corpus are id 0 (= OOV in the
32,768 LUT)? Sizes the error of feeding the Llama KD teacher inverse-LUT ids on the cloud (raw ids are gone there).
CPU only, 20 x 1M-token samples spread over the file."""
import numpy as np
d = np.memmap("/home/glenn/projects/bdh/data/tokenized_v4/train_lut32768_u16.bin", dtype=np.uint16, mode="r")
lut = np.load("/home/glenn/projects/neuro/data/vocab_map_32768.npy")
n = 0; z = 0
for s in np.linspace(0, len(d) - 1_000_001, 20).astype(np.int64):
    a = np.asarray(d[s:s + 1_000_000]); n += a.size; z += int((a == 0).sum())
print(f"[OK] id-0 fraction {z / n:.5f} ({z} of {n}) | LUT: {int((lut == 0).sum())} of {lut.size} raw ids map to 0, "
      f"max student id {int(lut.max())}")
