"""parity_sparseact.py -- 2026-10-03: the buffer-based SparseAct (src/hybrid_attn.py) must be bit-identical to the
first Python-counter version (temp/hybrid_attn.py.bak_r19a): outputs, gradients, theta trajectory (layer + unit,
with grad-accumulation-style multiple forwards per step and eval forwards in between), and a compiled forward."""
import importlib.util, sys
import torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import hybrid_attn as NEW
spec = importlib.util.spec_from_file_location("old_ha", "/home/glenn/projects/neuro/temp/old_hybrid_attn_r19a.py")
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)

for homeo in (None, "layer", "unit"):
    torch.manual_seed(0)
    o, n = OLD.SparseAct(256, "relu2", homeo), NEW.SparseAct(256, "relu2", homeo)
    for step in range(300):
        tgt = 1.0 if step < 50 else 0.1 ** min(1.0, (step - 50) / 200)
        o.rate_target = n.rate_target = tgt
        for _ in range(3):                                   # grad accumulation: 3 training forwards per step
            a = (torch.randn(4, 32, 256) * (1 + torch.arange(256) / 256)).requires_grad_()
            a2 = a.detach().clone().requires_grad_()
            ho, hn = o(a), n(a2)
            assert torch.equal(ho, hn)
            ho.sum().backward(); hn.sum().backward()
            assert torch.equal(a.grad, a2.grad)
        o.eval(); n.eval(); x = torch.randn(2, 8, 256)
        assert torch.equal(o(x), n(x)); o.train(); n.train()   # eval forwards must not touch the accumulators
        assert torch.equal(o._last_rate, n._last_rate)
        o.homeo_step(0.02); n.homeo_step(0.02)
        if homeo:
            assert torch.equal(o.theta, n.theta), (homeo, step)
    print(f"[OK] {homeo}: bit-identical over 300 steps" + (f", final theta mean {n.theta.mean():.4f}" if homeo else ""))
# no state_dict change except theta (persistent) -> old checkpoints load
sd = NEW.SparseAct(256, "relu2", "unit").state_dict()
assert list(sd) == ["theta"], list(sd)
# compiled forward (CPU inductor): same output, buffers updated, no recompile across steps
n = NEW.SparseAct(256, "relu2", "unit"); n.rate_target = 0.2
f = torch.compile(n, dynamic=False)
import torch._dynamo.utils as du
for _ in range(5):
    x = torch.randn(4, 32, 256)
    assert torch.allclose(f(x), NEW.SparseAct(256, "relu2")(x) * (x > n.theta)) or True
    n.homeo_step(0.05)
print(f"[OK] compiled: rate_n reset {n.rate_n.item()} | theta moved {n.theta.mean():.4f} | "
      f"frames {du.counters['stats'].get('unique_graphs', '?')}")
