"""Per-parameter-group breakdown of the v4 draft (gate_rank:16), to see which shape
knobs (d, n_neurons, d_mem, n_layers) drive the count before sizing a 1B v4.
CPU build only. Run: venv python temp/v4_1b_breakdown.py
"""
import os, sys, re, yaml, collections
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model

c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/flagship_v4_363m.draft.yaml"))
m = build_model(c)
tot = sum(p.numel() for p in m.parameters())
print(f"total {tot/1e6:.2f}M")
g = collections.Counter()
for n, p in m.named_parameters():
    key = re.sub(r"\.\d+\.", ".*.", n)  # collapse layer index
    g[key] += p.numel()
for k, v in g.most_common(40):
    print(f"{v/1e6:9.3f}M  {k}  ")
# one representative shape per group
seen = set()
for n, p in m.named_parameters():
    key = re.sub(r"\.\d+\.", ".*.", n)
    if key not in seen:
        seen.add(key); print(f"   {key}: {tuple(p.shape)}")
