"""diag_doc_lengths.py -- 2026-09-29: are documents delimited in the pretraining stream, and how long are they?
(Decides whether contiguous/stateful training has anything to carry across 512-token windows.) CPU only."""
import numpy as np, json
BIN = "/home/glenn/projects/bdh/data/tokenized/train.bin"
d = np.memmap(BIN, dtype=np.uint32, mode="r")
print(f"total tokens {len(d)/1e9:.2f}B")
meta = json.load(open("/home/glenn/projects/neuro/data/meta.json"))
print("meta keys:", list(meta)[:20])
rng = np.random.default_rng(0)
for name, tok in (("<|begin_of_text|>", 128000), ("<|end_of_text|>", 128001), ("<|eot_id|>", 128009)):
    n = 0; tot = 0
    for s in rng.integers(0, len(d) - 10_000_000, size=5):
        blk = np.asarray(d[s:s + 10_000_000]); n += int((blk == tok).sum()); tot += len(blk)
    print(f"{name} ({tok}): {n / tot * 1e6:.1f} per 1M tokens")
# doc-length distribution from the most frequent candidate separator
blk = np.asarray(d[1_000_000_000:1_050_000_000])
for tok in (128000, 128001):
    pos = np.flatnonzero(blk == tok)
    if len(pos) > 100:
        L = np.diff(pos)
        q = np.percentile(L, [10, 25, 50, 75, 90, 99])
        frac = [(L > t).mean() for t in (512, 1024, 2048, 4096)]
        tok_frac = [L[L > t].sum() / L.sum() for t in (512, 1024, 2048, 4096)]
        print(f"sep {tok}: docs={len(L)} mean={L.mean():.0f} pct10/25/50/75/90/99={q.astype(int).tolist()}")
        print(f"  frac docs longer than 512/1k/2k/4k: {[round(float(f),3) for f in frac]}")
        print(f"  frac TOKENS in docs longer than 512/1k/2k/4k: {[round(float(f),3) for f in tok_frac]}")
