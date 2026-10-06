# diff_sft_cfgs.py -- diff cfgs stored in the v1 SFT (recall kept) vs v3 SFT (recall lost) ckpts (2026-09-26)
import glob, torch
def cfg(run):
    ck = sorted(glob.glob(f"{run}/step_*.pt"))[-1]
    b = torch.load(ck, map_location="cpu", weights_only=False)
    extra = {k: v for k, v in b.items() if k not in ("model", "opt", "optimizer", "cfg") and not hasattr(v, "shape") and not isinstance(v, dict)}
    return ck, b.get("cfg", {}), extra
a_ck, a, ae = cfg("/home/glenn/neuro_trash_2026_09_25/checkpoints/sft_flagship_363m")
b_ck, b, be = cfg("models/checkpoints/sft_v3_363m")
print("A", a_ck, ae); print("B", b_ck, be)
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k):
        print(f"{k:32s} v1={a.get(k)!r:40.40s} v3={b.get(k)!r:.40s}")
