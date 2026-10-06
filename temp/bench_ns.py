"""bench_ns.py -- 2026-09-27: Muon Newton-Schulz cost per optimizer step, from the REAL Muon param
shapes (meta-device model build) for the v4 draft (363M-class) and the 1B A' shape. Times NS on the
5080 per unique shape and compares to model FLOPs per optimizer step (6*N*tokens estimate)."""
import sys, time, yaml, collections, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, _build_optimizers
from muon import _newton_schulz, Muon

def muon_shapes(c):
    with torch.device("meta"):
        m = build_model(c)
    opts = _build_optimizers(m, c, "meta")
    shapes = collections.Counter()
    for o in opts:
        if isinstance(o, Muon):
            for g in o.param_groups:
                for p in g["params"]:
                    shapes[tuple(p.shape)] += 1
    return shapes, sum(p.numel() for p in m.parameters())

def ns_ms(shape, n=10):
    G = torch.randn(*shape, device="cuda")
    for _ in range(2): _newton_schulz(G)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): _newton_schulz(G)
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3

def ns_flops(shape, steps=5):
    m, n = sorted(shape)
    return steps * (4 * m * m * n + 2 * m ** 3)

base = yaml.safe_load(open("/home/glenn/projects/neuro/configs/flagship_v4_363m.draft.yaml"))
cases = {
         "1B A' d2048/9L/N9728": dict(d=2048, d_mem=2048, n_neurons=9728, n_layers=9, attention_heads=16, batch_size=8, grad_accum=9)}
for name, ov in cases.items():
    c = dict(base); c.update(ov)
    shapes, nparams = muon_shapes(c)
    tot_ms, tot_fl, cnt = 0.0, 0, 0
    print(f"\n== {name}: {nparams/1e6:.0f}M params, {sum(shapes.values())} Muon matrices")
    for s, k in sorted(shapes.items(), key=lambda kv: -ns_flops(kv[0]) * kv[1]):
        ms = ns_ms(s); tot_ms += ms * k; tot_fl += ns_flops(s) * k; cnt += k
        print(f"  {str(s):16s} x{k:3d}: {ms:7.2f} ms each  ({ns_flops(s)/1e9:7.1f} GFLOP)")
    tok_step = c["batch_size"] * c["grad_accum"] * c["block_size"]
    model_fl = 6 * nparams * tok_step
    print(f"  NS total per optimizer step: {tot_ms:.0f} ms on 5080, {tot_fl/1e12:.1f} TFLOP"
          f" vs model ~{model_fl/1e12:.0f} TFLOP/step (6N x {tok_step} tok) -> NS = {100*tot_fl/model_fl:.1f}% extra FLOPs")
