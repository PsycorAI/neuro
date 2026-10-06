"""diag_fpt_memory.py -- 2026-09-27: how much temporal memory does the trained K=10 FPT neuron use?
On real currents from the trained 30M model, compare FPT spikes with (a) a memoryless threshold
H(cur_t - theta) (best theta), (b) exact LIF, (c) exact no-reset. Also: does perturbing cur at t-1..t-8
change FPT spikes at t (temporal influence)?"""
import sys, yaml, numpy as np, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN
import lif_parallel as LP
from lif_exact import lif_exact
dev = "cuda"
c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/screen_indepwin_s0_30m.yaml"))
m = build_model(c).to(dev).eval()
m.load_state_dict(torch.load("/home/glenn/projects/neuro/models/checkpoints/screen_indepwin_s0_30m/step_0006104.pt",
                             map_location=dev, weights_only=False)["model"], strict=False)
data = StreamText(BIN, vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256))
curs = []; FPT = LP.lif_parallel
@torch.compiler.disable
def grab(cur, beta, vth=1.0, K=3, alpha=2.0):
    curs.append((cur.detach().float().clone(), beta, K)); return FPT(cur, beta, vth=vth, K=K, alpha=alpha)
LP.lif_parallel = grab
np.random.seed(7)
with torch.no_grad():
    x, _ = data.batch(4, c["block_size"], "val", dev)
    with torch.autocast(dev, dtype=torch.bfloat16):
        m(x)
LP.lif_parallel = FPT
for li, (cur, beta, K) in enumerate(curs):
    s = FPT(cur, beta, K=K)[0].round()
    rate = s.mean().item()
    best = max(((s == (cur > th).float()).float().mean().item(), th) for th in np.linspace(0.0, 2.5, 51))
    ex1 = (lif_exact(cur, beta, gamma=1.0)[0].round() == s).float().mean().item()
    ex0 = (lif_exact(cur, beta, gamma=0.0)[0].round() == s).float().mean().item()
    # temporal influence: perturb cur at one position t0, count spike changes at t0 and later
    t0 = 200; cp = cur.clone(); cp[:, t0] += 0.5 * cur.std()
    sp = FPT(cp, beta, K=K)[0].round()
    d = (sp != s).float().mean(dim=(0, 2))           # (T,)
    print(f"layer {li}: K={K} rate {rate:.3f} | agree: memoryless H(cur>{best[1]:.2f}) {best[0]:.3f}, "
          f"exact LIF {ex1:.3f}, exact no-reset {ex0:.3f} | perturb@{t0}: flips at t0 {d[t0]:.3f}, "
          f"t0+1..8 {d[t0+1:t0+9].mean():.3f}, t0+9..64 {d[t0+9:t0+65].mean():.4f}, before t0 {d[:t0].mean():.4f}")
    # beta-scale of cur: how big is cur relative to threshold
    print(f"          cur mean {cur.mean():.3f} std {cur.std():.3f} | frac cur>1: {(cur>1).float().mean():.3f}")
