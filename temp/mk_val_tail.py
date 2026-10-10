"""mk_val_tail.py -- 2026-10-09 (Neuro 54). The v4 val split = the LAST 14,527,194 tokens of the uint16 corpus, i.e. the
end of piece 021. A run that starts on a PARTIAL corpus needs it as its own small file (29 MB):
cloud/pull_corpus.sh joins the available pieces + this tail, so data_val_tokens and every val window stay the same.
  -> G:\\neuro_corpus_v4\\val_tail_u16.bin (+ .sha256)"""
import hashlib, numpy as np
SRC = "/home/glenn/projects/bdh/data/tokenized_v4/train_lut32768_u16.bin"
DST = "/mnt/g/neuro_corpus_v4/val_tail_u16.bin"
VAL = 14_527_194
b = np.asarray(np.memmap(SRC, dtype=np.uint16, mode="r")[-VAL:]).tobytes()
open(DST, "wb").write(b)
open(DST + ".sha256", "w").write(f"{hashlib.sha256(b).hexdigest()}  val_tail_u16.bin\n")
print(f"[OK] {DST}: {VAL:,} tokens, {len(b) / 2**20:.1f} MiB")
