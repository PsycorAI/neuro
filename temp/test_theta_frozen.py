# CPU check (2026-10-07): spike_theta_frozen -> optimizers build, one step runs, thetas unchanged, other weights move.
import sys, yaml, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, _build_optimizers
from spike_code import SpikeCode
c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/screen_p26k_7L_spk_kd_qv_sc20.yaml"))
c.update(lif_exact=False, lif_signed=False, lif_levels=1, lif_learn_beta=False, compile=False, amp=False,
         spike_theta_frozen=True)
m = build_model(c)
for _m in m.modules():
    if isinstance(_m, SpikeCode):
        _m.theta.requires_grad_(False)
opts = _build_optimizers(m, c, "cpu")
th0 = [x.theta.clone() for x in m.modules() if isinstance(x, SpikeCode)]
w0 = m.head.weight.clone()
x = torch.randint(0, c["vocab"], (1, 64))
m(x).float().logsumexp(-1).mean().backward()
for o in opts:
    o.step()
th1 = [x.theta for x in m.modules() if isinstance(x, SpikeCode)]
assert all(torch.equal(a, b) for a, b in zip(th0, th1)), "theta moved"
assert not torch.equal(w0, m.head.weight), "head did not move"
print(f"[OK] spike_theta_frozen: {len(th0)} thetas unchanged after a step, other weights updated")
