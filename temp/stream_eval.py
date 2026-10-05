"""stream_eval.py -- 2026-09-29: streaming (carried-state) eval vs the standard zero-state eval, by position in window.
The val tail is split into S contiguous sub-streams, each read as consecutive 512-token windows (batched over S).
  zero  : every window starts from zero state (= how val_ppl is measured today)
  carry : GDN memory M + prev_spk carried across windows (detached); attention stays window-local
Reports overall ppl and ppl by position bucket [0,16) [16,64) [64,256) [256,512), plus the window-0 (cold) vs
windows>=1 split for carry. Eval-only. Usage: CUDA_VISIBLE_DEVICES=1 python temp/stream_eval.py name=ckpt [...]"""
import sys, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, BIN
S, W, NWIN = 16, 512, 36                       # 16 streams x 36 windows x 512 = 295k val tokens
BUCKETS = [(0, 16), (16, 64), (64, 256), (256, 512)]
torch.set_float32_matmul_precision("high")

def streams(vocab):
    d = np.memmap(BIN, dtype=np.uint32, mode="r")
    lut = np.load(f"/home/glenn/projects/neuro/data/vocab_map_{vocab}.npy")
    lo = int(len(d) * 0.999); n = S * (NWIN * W + 1)
    return lut[np.asarray(d[lo:lo + n], dtype=np.int64)].reshape(S, NWIN * W + 1)

@torch.no_grad()
def run(m, toks, carry_on):
    ce_pos = torch.zeros(W); n_pos = 0; ce_w0 = ce_rest = 0.0; carry = None
    for i in range(NWIN):
        x = torch.from_numpy(toks[:, i * W:(i + 1) * W]).cuda()
        y = torch.from_numpy(toks[:, i * W + 1:(i + 1) * W + 1]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if carry_on:
                out, carry = m(x, initial_state=carry, return_final_state=True)
            else:
                out = m(x)
        ce = F.cross_entropy(out.float().reshape(-1, out.size(-1)), y.reshape(-1), reduction="none").view(S, W).cpu()
        ce_pos += ce.sum(0); n_pos += S
        if i == 0: ce_w0 += ce.mean().item()
        else: ce_rest += ce.mean().item() / (NWIN - 1)
    ce_pos /= n_pos
    return (math.exp(ce_pos.mean().item()), [math.exp(ce_pos[a:b].mean().item()) for a, b in BUCKETS],
            math.exp(ce_w0), math.exp(ce_rest))

for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    toks = streams(c["vocab"])
    z, zb, _, _ = run(m, toks, False)
    k, kb, k0, kr = run(m, toks, True)
    print(f"{name:12s} zero {z:6.1f} | carry {k:6.1f} ({(k - z) / z * 100:+5.1f}%) | by pos bucket zero->carry: "
          + "  ".join(f"{a}-{b}: {p:.0f}->{q:.0f}" for (a, b), p, q in zip(BUCKETS, zb, kb))
          + f" | carry win0 {k0:.0f} vs win>=1 {kr:.0f}", flush=True)
    del m; torch.cuda.empty_cache()
