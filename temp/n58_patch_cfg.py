"""n58_patch_cfg.py <ckpt> -- 2026-10-12 (Neuro 58). A cloud checkpoint's stored cfg points data_bin at the instance
(/workspace/data/...); the home eval scripts open cfg["data_bin"]. Rewrites that one key to the home corpus (same file,
same val tail) and saves in place (tmp + atomic replace). Weights and optimizer state are untouched."""
import os, sys, torch
p = sys.argv[1]; HOME = "/home/glenn/projects/bdh/data/tokenized_v4/train_lut32768_u16.bin"
s = torch.load(p, map_location="cpu", weights_only=False)
old = s["cfg"].get("data_bin")
if old == HOME and s["cfg"].get("data_prelut", False):
    print("cfg already points home"); sys.exit(0)
s["cfg"]["data_bin"] = HOME; s["cfg"]["data_prelut"] = True   # n61 10-14: u16 LUT file needs prelut (conversion cfgs: False)
torch.save(s, p + ".tmp"); os.replace(p + ".tmp", p)
print(f"patched data_bin: {old} -> {HOME} | step {s.get('step')} | tokens {s.get('tokens')}")
