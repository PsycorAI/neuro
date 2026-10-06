"""test_muon_batched.py -- (a) default Muon path == HEAD muon.py bit-for-bit; (b) batched_ns path vs
per-matrix; (c) interleaved optimizer-step timing, real Muon shapes (v4 draft + 1B A')."""
import sys, time, yaml, importlib.util, subprocess, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model
import muon as M_new
src = subprocess.run(["git", "-C", "/home/glenn/projects/neuro", "show", "HEAD:src/muon.py"],
                     capture_output=True, text=True, check=True).stdout
open("/tmp/muon_head.py", "w").write(src)
spec = importlib.util.spec_from_file_location("muon_head", "/tmp/muon_head.py")
M_old = importlib.util.module_from_spec(spec); spec.loader.exec_module(M_old)

def muon_shapes(c):
    with torch.device("meta"):
        m = build_model(c)
    mu, *_ = M_new.build_muon_adamw(m, fix_gate_matrices=c.get("muon_fix_gate_matrices", False),
                                    route_embeddings_by_type=c.get("muon_route_embeddings_by_type", False),
                                    head_weight_decay=c.get("head_weight_decay"))
    return [tuple(p.shape) for g in mu.param_groups for p in g["params"]]

def make(shapes, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(*s, device="cuda", generator=g) * 0.02) for s in shapes]

def run(opt_cls, shapes, steps=3, **kw):
    ps = make(shapes); opt = opt_cls(ps, lr=0.005, weight_decay=0.1, normuon=True, **kw)
    for t in range(steps):
        g = torch.Generator(device="cuda").manual_seed(100 + t)
        for p in ps: p.grad = torch.randn(p.shape, device="cuda", generator=g)
        opt.step()
    return ps, opt

base = yaml.safe_load(open("/home/glenn/projects/neuro/configs/flagship_v4_363m.draft.yaml"))
cases = {"v4 draft": {}, "1B A'": dict(d=2048, d_mem=2048, n_neurons=9728, n_layers=9, attention_heads=16)}
for name, ov in cases.items():
    c = dict(base); c.update(ov)
    shapes = muon_shapes(c)
    small = [s for s in shapes if s[0] * s[1] <= M_new.BATCHED_NS_MAX_NUMEL]
    print(f"\n== {name}: {len(shapes)} Muon matrices, {len(small)} small (batched)")
    if name == "v4 draft":
        po, _ = run(M_old.Muon, shapes); pn, _ = run(M_new.Muon, shapes)
        print(f"(a) default path bit-identical to HEAD: {all(torch.equal(a, b) for a, b in zip(po, pn))}")
        pb, _ = run(M_new.Muon, shapes, batched_ns=True)
        worst = max(((a - b).abs().max() / (a.abs().max())).item() for a, b in zip(pn, pb))
        exact = sum(torch.equal(a, b) for a, b in zip(pn, pb))
        print(f"(b) batched vs per-matrix after 3 steps: {exact}/{len(pn)} params bit-identical, worst rel diff {worst:.2e}")
        del po, pn, pb
    # (c) timing, interleaved: warm both, then alternate
    ps = make(shapes)
    opts = {"per-matrix": M_new.Muon(ps, lr=0.005, weight_decay=0.1, normuon=True),
            "batched": M_new.Muon(ps, lr=0.005, weight_decay=0.1, normuon=True, batched_ns=True)}
    for p in ps: p.grad = torch.randn_like(p)
    for o in opts.values(): o.step(); o.step()
    ts = {k: [] for k in opts}
    for r in range(8):
        for k, o in opts.items():
            torch.cuda.synchronize(); t = time.time(); o.step(); torch.cuda.synchronize()
            ts[k].append((time.time() - t) * 1e3)
    for k, v in ts.items():
        v.sort(); print(f"(c) {k:10s}: median {v[len(v)//2]:6.1f} ms  (min {v[0]:.1f}, max {v[-1]:.1f})")
    del ps, opts; torch.cuda.empty_cache()
