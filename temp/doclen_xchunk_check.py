# How much cross-chunk (same-document) content does a T1024 / chunk-512 window carry?
# CPU-only research check (Neuro 52, 10-06). Reads the raw uint32 stream to find the doc separator.
import numpy as np
raw = np.memmap('/home/glenn/projects/bdh/data/tokenized_v4/train.bin', dtype=np.uint32, mode='r')
N = len(raw)
rng = np.random.default_rng(0)
# find separator: most common token among candidates 128000/128001
s = np.asarray(raw[:20_000_000])
for t in (128000, 128001, 128009):
    print('token', t, 'count in first 20M:', int((s == t).sum()))
cnt = {t: int((s == t).sum()) for t in (128000, 128001)}
sep = max(cnt, key=cnt.get)
print('separator =', sep)
for name, lo, hi in (('old part', 0, 14_512_665_894), ('fwe top-up', 14_512_665_894, 14_512_665_894 + 8_624_395_115)):
    starts = rng.integers(lo, hi - 1024, size=4000)
    same_doc_frac = []      # chunk-2 tokens whose doc began before chunk 2 (memory could help)
    any_cross = 0           # windows where chunk 2 starts mid-doc continuing from chunk 1
    long_carry = []         # chunk-2 tokens whose doc began >= 512 tokens before them
    doclens = []
    for st in starts:
        w = np.asarray(raw[st:st + 1024])
        pos = np.where(w == sep)[0]
        # last separator in chunk 1 (or none) -> doc start of the doc running at position 512
        b = pos[pos < 512]
        doc_start_at_512 = b[-1] if len(b) else -1
        nxt = pos[pos >= 512]
        end = nxt[0] if len(nxt) else 1024
        n_carry = end - 512  # chunk-2 tokens in the same doc as chunk 1 content
        cross = n_carry > 0 and doc_start_at_512 < 511
        any_cross += int(cross)
        same_doc_frac.append(n_carry / 512 if cross else 0.0)
        if len(pos) >= 2:
            doclens.extend(np.diff(pos).tolist())
    print(f'{name}: windows with chunk2 continuing a chunk1 doc {any_cross/len(starts):.1%} | '
          f'mean chunk2 share in a chunk1 doc {np.mean(same_doc_frac):.1%} | '
          f'doc len median {np.median(doclens):.0f} mean {np.mean(doclens):.0f} (n {len(doclens)})')
