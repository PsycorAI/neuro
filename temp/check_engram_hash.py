# 2026-10-01 code review: does EngramV2's low-bit hash alias token ids that differ by K in EVERY table?
import sys, torch
sys.path.insert(0, '/home/glenn/projects/neuro/src')
from engram_v2 import EngramV2
for V, K in ((16384, 8192), (32768, 65536), (32768, 8192)):
    m = EngramV2(d=8, vocab=V, orders=(2, 3), K=K, heads=4, rank=4)
    a = torch.randint(0, V, (1, 4096)); b = a.clone()
    hi = b >= (V // 2); b[hi] = b[hi] - V // 2          # flip the top id bit
    same = []
    for t, n in enumerate([2] * 4 + [3] * 4):
        same.append((m._hash_order_head(a, n, t) == m._hash_order_head(b, n, t)).float().mean().item())
    print(f'V={V} K={K}: share of positions whose bucket is unchanged after flipping the top id bit, per table:',
          [round(s, 3) for s in same])
