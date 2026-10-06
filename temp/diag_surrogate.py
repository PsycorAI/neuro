"""diag_surrogate.py -- mean surrogate-gradient factor g'(pre-vth)=1/(1+(2pi*over)^2) and |over| stats
per neuron variant, on the trained 30M model's real currents (bf16 as in training)."""
import sys, math, yaml, numpy as np, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN
import lif_parallel as LP
dev = "cuda"
c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/screen_indepwin_s0_30m.yaml"))
m = build_model(c).to(dev).eval()
m.load_state_dict(torch.load("/home/glenn/projects/neuro/models/checkpoints/screen_indepwin_s0_30m/step_0006104.pt",
                             map_location=dev, weights_only=False)["model"], strict=False)
data = StreamText(BIN, vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256))
curs = []; FPT = LP.lif_parallel
@torch.compiler.disable
def grab(cur, beta, vth=1.0, K=3, alpha=2.0):
    curs.append(cur.detach().clone()); return FPT(cur, beta, vth=vth, K=K, alpha=alpha)
LP.lif_parallel = grab
np.random.seed(7)
with torch.no_grad():
    x, _ = data.batch(4, c["block_size"], "val", dev)
    with torch.autocast(dev, dtype=torch.bfloat16):
        m(x)
LP.lif_parallel = FPT
def pre_of(cur, K=None, gamma=None, beta=0.9):
    if K is not None:                     # FPT final pre with s^K resets (bf16 like training)
        s = torch.zeros_like(cur)
        for _ in range(K):
            sp = torch.zeros_like(s); sp[:, 1:] = s[:, :-1]
            s = (LP._scan_chunked(beta, cur - beta * sp) > 1.0).to(cur.dtype)
        sp = torch.zeros_like(s); sp[:, 1:] = s[:, :-1]
        return LP._scan_chunked(beta, cur - beta * sp).float()
    c32 = cur.float(); mem = torch.zeros_like(c32[:, 0]); out = []
    for t in range(c32.shape[1]):
        pre = beta * mem + c32[:, t]; s = (pre > 1.0).float(); out.append(pre); mem = pre - gamma * s
    return torch.stack(out, 1)
for li, cur in enumerate(curs):
    print(f"layer {li}:")
    for name, kw in [("FPT K=10", dict(K=10)), ("FPT K=3", dict(K=3)), ("FPT K=0", dict(K=0)),
                     ("exact LIF g=1", dict(gamma=1.0)), ("exact no-reset g=0", dict(gamma=0.0))]:
        over = pre_of(cur, **kw) - 1.0
        gp = 1.0 / (1.0 + (2 * math.pi * over) ** 2)
        print(f"  {name:20s} rate {(over > 0).float().mean():.3f} | mean g' {gp.mean():.4f} | "
              f"frac |over|<0.1 {(over.abs() < 0.1).float().mean():.3f} | median |over| {over.abs().median():.3f}")
