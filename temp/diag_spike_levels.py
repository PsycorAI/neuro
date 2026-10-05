"""diag_spike_levels.py -- 2026-09-30: what is the 'spike rate'? Reported rate = mean |level| (a level-4 spike counts 4).
Measures, on real val text, for each spiking block: mean |s| (reported), fraction of (neuron, token) that fire at all
(= events on graded-spike hardware, e.g. Loihi 2), the level histogram, and the fraction of neurons that never fire
(dead) / fire on >50% of tokens (saturated). Eval only. Usage: python temp/diag_spike_levels.py name=ckpt [...]"""
import sys, numpy as np, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    cap = []
    for b in m.blocks:
        if hasattr(b, "qkv"): continue
        orig = b.forward_sequence
        def fs(*a, _o=orig, **k):
            out = _o(*a, **k); cap.append(out[3].detach().float()); return out
        b.forward_sequence = fs
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256))
    np.random.seed(0); x, _ = data.batch(16, 512, "val", "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m(x)
    for i, sp in enumerate(cap):
        a = sp.abs(); nz = (a > 0).float()
        per_neuron = nz.mean((0, 1))
        hist = {int(v): f"{(a.round() == v).float().mean().item():.3f}" for v in range(1, int(a.max().item()) + 1)}
        print(f"{name:12s} blk{i}: mean|s| {a.mean():.3f} | fire frac {nz.mean():.3f} | mean level when firing "
              f"{(a.sum() / nz.sum().clamp_min(1)):.2f} | level frac {hist} | dead {(per_neuron == 0).float().mean():.1%} "
              f"| >50% {(per_neuron > 0.5).float().mean():.1%}", flush=True)
    del m; torch.cuda.empty_cache()
