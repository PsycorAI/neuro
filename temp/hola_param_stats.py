# HOLA parameter stats from a ckpt (CPU, no model build): how much did the cache read get used?
import sys, torch
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    for k in sorted(k for k in sd if "hola_" in k):
        t = sd[k].float()
        print(f"{name:8s} {k:28s} mean {t.mean():+.4f} |mean| {t.abs().mean():.4f} max|x| {t.abs().max():.4f}")
