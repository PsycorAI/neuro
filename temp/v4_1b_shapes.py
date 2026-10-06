"""Exact param counts for candidate 1B v4 shapes (v4 draft recipe, gate_rank:16).
CPU build only (safe while the 5090 trains). Also a rough FLOPs/tok (6 x non-embed
matmul params + head) and the tok/s needed for 16B tokens in 30 days.
Run: venv python temp/v4_1b_shapes.py
"""
import sys, yaml, gc
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model

base = yaml.safe_load(open("/home/glenn/projects/neuro/configs/flagship_v4_363m.draft.yaml"))
# name, d, d_mem, n_layers, attention positions, n_neurons
CANDS = [
    ("v4draft-ref",   1280, 1280,  9, [4],     3840),
    ("A wide-d2048",  2048, 2048,  9, [4],     9216),   # 4.5xd
    ("A' wide-d2048", 2048, 2048,  9, [4],     9728),   # 4.75xd
    ("B deep-d1792",  1792, 1792, 12, [4, 10], 7168),   # 4.0xd, 12 layers
    ("C d2048-mem1280", 2048, 1280, 9, [4],   12288),   # 6.0xd, small state/big width
]
need = 16e9 / (30 * 86400)
print(f"tok/s needed for 16B in 30d: {need:,.0f}")
print(f"{'shape':18s} {'d':>5} {'dmem':>5} {'L':>3} {'N':>6} {'N/d':>5} {'total':>9} {'non-emb':>9} {'GF/tok':>7} {'MFU@need':>8}")
for name, d, dm, L, attn, N in CANDS:
    c = dict(base); c.update(d=d, d_mem=dm, n_layers=L, attention_layer_positions=attn,
                             attention_heads=d // 128, n_neurons=N)
    m = build_model(c)
    tot = sum(p.numel() for p in m.parameters())
    emb = m.embed.weight.numel() + m.head.weight.numel()
    # training FLOPs/tok ~ 6 x (matmul params excluding the embedding lookup)
    gf = 6 * (tot - m.embed.weight.numel()) / 1e9
    mfu = need * gf * 1e9 / 104.8e12
    print(f"{name:18s} {d:5d} {dm:5d} {L:3d} {N:6d} {N/d:5.2f} {tot/1e6:8.1f}M {(tot-emb)/1e6:8.1f}M {gf:7.2f} {mfu*100:7.1f}%")
    del m; gc.collect()
