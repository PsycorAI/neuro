import sys, yaml, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model
for n in ["probe_g4_c0s1", "probe_g4_tfmrs1"]:
    c = yaml.safe_load(open(f"/home/glenn/projects/neuro/configs/{n}.yaml"))
    m = build_model(c)   # CPU fp32 (~4 GB); meta device fails on init-time .item()
    tot = sum(p.numel() for p in m.parameters())
    emb = sum(p.numel() for nm, p in m.named_parameters() if "emb" in nm.lower() or "head" in nm.lower())
    print(f"{n}: total {tot/1e6:.0f}M | embed/head/engram-ish {emb/1e6:.0f}M | non-emb {(tot-emb)/1e6:.0f}M")
