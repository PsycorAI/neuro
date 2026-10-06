"""check_model_leak.py -- 2026-09-27: model-level future-token leak from _scan_chunked's bf16 arange.
For a trained checkpoint: (1) causality: change input token at p, do logits at positions < p move?
(2) val loss with the stock scan vs a leak-free scan (fp32 index math), split t<256 / t>=256."""
import sys, math, yaml, glob, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN
import lif_parallel as LP
dev = "cuda"
STOCK = LP._scan_chunked
def scan_fixed(beta, x, chunk=512):
    """Same math as _scan_chunked but indices/decay built in fp32 (no bf16 integer rounding)."""
    B, T, N = x.shape
    out = torch.empty_like(x); carry = torch.zeros(B, N, device=x.device, dtype=x.dtype)
    for start in range(0, T, chunk):
        end = min(start + chunk, T); L = end - start
        idx = torch.arange(L, device=x.device, dtype=torch.float32)
        diff = idx[:, None] - idx[None, :]
        W = torch.where(diff >= 0, beta ** diff.clamp(min=0), torch.zeros_like(diff)).to(x.dtype)
        y = torch.matmul(W, x[:, start:end]) + (beta ** (idx + 1)).to(x.dtype).view(1, L, 1) * carry.unsqueeze(1)
        out[:, start:end] = y; carry = y[:, -1]
    return out

def run(cfg, ck, B=4, iters=10):
    c = yaml.safe_load(open(cfg))
    m = build_model(c).to(dev).eval()
    m.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model"], strict=False)
    data = StreamText(BIN, vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256))
    T = c["block_size"]
    def fwd(x):
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
            out = m(x)
        return (out[0] if isinstance(out, tuple) else out).float()
    print(f"\n== {ck.split('/')[-2]}  ({sum(p.numel() for p in m.parameters())/1e6:.0f}M)")
    np.random.seed(3); x, y = data.batch(B, T, "val", dev)
    for name, scan in (("stock bf16 scan", STOCK), ("leak-free scan", scan_fixed)):
        LP._scan_chunked = scan
        l0 = fwd(x)
        msg = []
        for p in (300, 420):
            xp = x.clone(); xp[:, p] = (xp[:, p] + 1234) % c["vocab"]
            d = (fwd(xp) - l0).abs().amax(dim=(0, 2))
            msg.append(f"token@{p} moves logits before it by {d[:p].max().item():.3g} (at/after: {d[p:].max().item():.3g})")
        print(f"  [{name}] causality: " + " | ".join(msg))
    for name, scan in (("stock bf16 scan", STOCK), ("leak-free scan", scan_fixed)):
        LP._scan_chunked = scan
        np.random.seed(1234); lo = hi = 0.0
        for _ in range(iters):
            x, y = data.batch(B, T, "val", dev)
            ce = F.cross_entropy(fwd(x).reshape(-1, c["vocab"]), y.reshape(-1), reduction="none").view(B, T)
            lo += ce[:, :256].mean().item(); hi += ce[:, 256:].mean().item()
        lo /= iters; hi /= iters
        print(f"  [{name}] val ppl t<256: {math.exp(lo):8.2f} | t>=256: {math.exp(hi):8.2f} | all: {math.exp((lo+hi)/2):8.2f}")
    LP._scan_chunked = STOCK
    del m; torch.cuda.empty_cache()

N = "/home/glenn/projects/neuro"
run(f"{N}/configs/screen_indepwin_s0_30m.yaml", f"{N}/models/checkpoints/screen_indepwin_s0_30m/step_0006104.pt")
v3 = sorted(glob.glob(f"{N}/models/checkpoints/flagship_v3_363m/step_*.pt"))[-1]
run(f"{N}/configs/flagship_v3_363m.yaml", v3, B=2)
