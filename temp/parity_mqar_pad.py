"""parity_mqar_pad.py -- 2026-10-02 speed plan (compile fix). CPU, real c0 checkpoint.
(1) mqar_pad_to: right-padding an MQAR batch must not change the logits at the query position, nor the
    gradients of the MQAR loss (every layer causal). Tested at several N, train mode, DSN k mid-ramp.
(2) _kwta helper == the old inline k-WTA code (bitwise), several k.
LIF Triton kernels are swapped for the torch reference (as scripts/interp_scorecard.py)."""
import sys, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
sys.path.insert(0, "/home/glenn/projects/neuro/scripts")
import lif_exact as LE
from interp_scorecard import lif_ml_ref, load
LE.lif_exact_ml = lif_ml_ref
LE.lif_exact = lambda cur, beta, vth=1.0, K=None, alpha=2.0, gamma=1.0: lif_ml_ref(cur, beta, vth, gamma=gamma)
from train_gpu import mqar_aug_batch
from hybrid_attn import _kwta

torch.set_num_threads(4)
ok = True
# (2) k-WTA helper vs the old inline code
class F_: pass
f = F_(); f.k = 256
for kc in [None, 2752, 1000, 300, 256]:
    f.k_cur = kc
    a = torch.randn(2, 7, 2752)
    k = f.k if kc is None else min(int(kc), a.shape[-1])
    old = a
    if k < a.shape[-1]:
        thr = a.detach().abs().topk(k, dim=-1).values[..., -1:]
        old = a * (a.detach().abs() >= thr)
    same = torch.equal(old, _kwta(f, a))
    ok &= same
    print(f"[kwta] k_cur={kc}: bitwise equal = {same}")

# (1) MQAR right-padding
m, c = load("/home/glenn/projects/neuro/models/checkpoints/screen_r16_c0_s0_30m/step_0006104.pt", "cpu")
m.train()
for b in m.blocks:
    if hasattr(getattr(b, "ffn", None), "k_cur"):
        b.ffn.k_cur = 700                                  # mid-ramp k, exercises the k-WTA path
PAD = 2 * c.get("mqar_N_max", 32) + 1
params = [p for p in m.parameters() if p.requires_grad]
for N in [4, 17, 32]:
    torch.manual_seed(N)
    x, tg = mqar_aug_batch(4, N, "cpu")
    L = x.shape[1]
    out = []
    for xx in [x, F.pad(x, (0, PAD - L))]:
        m.zero_grad(set_to_none=True)
        o = m(xx)
        lo = (o[0] if isinstance(o, tuple) else o)[:, L - 1, :]
        F.cross_entropy(lo.float(), tg).backward()
        out.append((lo.detach().clone(), [p.grad.clone() if p.grad is not None else None for p in params]))
    (l0, g0), (l1, g1) = out
    dl = (l0 - l1).abs().max().item()
    dg = max(((a - b).abs().max() / (a.abs().max() + 1e-12)).item() for a, b in zip(g0, g1) if a is not None)
    good = dl < 1e-4 and dg < 1e-4
    ok &= good
    print(f"[pad] N={N} L={L}->{PAD}: max|dlogit|={dl:.2e} max rel dgrad={dg:.2e} -> {'OK' if good else 'FAIL'}")
print("ALL OK" if ok else "FAILED")
