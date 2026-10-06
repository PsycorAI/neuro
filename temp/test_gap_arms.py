"""test_gap_arms.py -- 2026-09-28: plumbing test for the ppl-gap flags. For each arm: load the control's weights into
the arm model (new params keep their init) and compare eval logits with the control (identity-init arms must match
exactly), then 3 train steps and check every NEW parameter receives a non-zero gradient."""
import sys, os, yaml, torch, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
if os.environ.get("GAP_STAGE"): sys.path.insert(0, os.environ["GAP_STAGE"])
from train_gpu import build_model, _build_optimizers
import model as M; print("model module:", M.__file__)
base = yaml.safe_load(open("/home/glenn/projects/neuro/configs/screen_honest_lifexact_s0_30m.yaml"))
ARMS = {"B1 ff_conv": dict(ff_conv=4), "B2 swa": dict(swa_window=64), "D2 out_gate": dict(mem_out_gate=16),
        "C2 learn_beta": dict(lif_learn_beta=True), "D1 pre_norm": dict(pre_norm=True, final_norm=True),
        "C1 levels4": dict(lif_levels=4), "C3 ternary": dict(lif_signed=True)}
torch.manual_seed(0)
ctl = build_model(base).to("cuda").eval()
g = torch.Generator(device="cuda").manual_seed(1)
x = torch.randint(0, base["vocab"], (4, 513), device="cuda", generator=g)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    ref = ctl(x[:, :-1]).float()
for name, flags in ARMS.items():
    c = dict(base, **flags)
    m = build_model(c).to("cuda")
    missing, unexpected = m.load_state_dict(ctl.state_dict(), strict=False)
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lg = m(x[:, :-1]).float()
    d = (lg - ref).abs().max().item()
    m.train(); opts = _build_optimizers(m, c, "cuda")
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, sr = m(x[:, :-1], return_stats=True)
            loss = F.cross_entropy(out.float().reshape(-1, c["vocab"]), x[:, 1:].reshape(-1)) + 0.5 * sr
        loss.backward()
        gnorm = {k: (p.grad.norm().item() if p.grad is not None else None) for k, p in m.named_parameters() if k in missing}
        [o.step() for o in opts]; [o.zero_grad() for o in opts]
    dead = [k for k, v in gnorm.items() if not v]
    print(f"{name:14s} new params {len(missing):2d} {sorted(set(k.split('.', 2)[-1] for k in missing))} | "
          f"init logits max|diff| vs control {d:.3e} | loss {loss.item():.3f} | spike_rate {sr.item():.3f} | "
          f"zero/None-grad new params: {dead or 'none'}")
    del m, opts; torch.cuda.empty_cache()
