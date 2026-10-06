"""Check that re-tokenizing decoded old-corpus docs reproduces their first-24-token hashes (validates dedup)."""
import sys, time, numpy as np
sys.path.insert(0, "/home/glenn/projects/neuro/scripts")
import build_corpus_v4 as B
tok = B.load_tok()
d = np.memmap(B.OLD, dtype=np.uint32, mode="r")
s = np.asarray(d[5_000_000_000:5_000_400_000]).astype(np.int64)
pos = np.flatnonzero(s == B.BOS)[:-1]
match = tot = 0
docs = []
for a, b in zip(pos[:200], pos[1:201]):
    ids = s[a + 1:b].tolist()
    if b - a < 40:
        continue
    docs.append(tok.decode(ids))
    re_ids = B.encode(tok, [docs[-1]])[0]
    W1 = np.zeros((1, B.L), np.uint64); W1[0, :min(B.L, len(ids))] = ids[:B.L]
    W2 = np.zeros((1, B.L), np.uint64); W2[0, :min(B.L, len(re_ids))] = re_ids[:B.L]
    tot += 1; match += int(B.hash_rows(W1)[0] == B.hash_rows(W2)[0])
print(f"re-tokenized docs whose 24-token hash matches the stored ids: {match}/{tot}")
print("sample decode:", repr(docs[3][:160]))
t0 = time.time(); n = sum(len(x) for x in B.encode(tok, docs * 20)); dt = time.time() - t0
print(f"tokenize speed: {n / dt / 1e6:.2f}M tok/s ({n:,} tokens)")
