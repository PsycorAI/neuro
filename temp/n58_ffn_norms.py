"""n58_ffn_norms.py <ckpt> [...] -- 2026-10-12 (Neuro 58). CPU, state dict only: per attention block the shapes and RMS of
the FFN tensors (and their biases), to see what a dead FFN looks like in the weights. Also lists the optimizer-state
layout (how many tensors each optimizer holds), needed for a re-initialisation at a resume."""
import sys, torch
for path in sys.argv[1:]:
    s = torch.load(path, map_location="cpu", weights_only=False); sd = s["model"]
    print(path, "| step", s.get("step"))
    for k, v in sd.items():
        if ".ffn." in k and torch.is_tensor(v) and v.is_floating_point():
            v = v.float()
            extra = f" | row-norm min {v.norm(dim=1).min():.4f} max {v.norm(dim=1).max():.3f}" if v.dim() == 2 else f" | mean {v.mean():+.4f}"
            print(f"  {k:34s} {str(tuple(v.shape)):16s} rms {v.pow(2).mean().sqrt():.5f}{extra}")
    for i, st in enumerate(s.get("opt_states", [])):
        print(f"  opt {i}: groups {[len(g['params']) for g in st['param_groups']]} | state entries {len(st['state'])} "
              f"| keys of one entry {sorted(next(iter(st['state'].values())).keys())}")
    del s
