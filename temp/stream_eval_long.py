"""stream_eval_long.py -- 2026-09-30: LONG-stream stability of carried state (the length-generalization claim of
state passing / TBTT, arXiv 2507.02782). 16 contiguous val streams x 120 windows x 512 = 983k tokens, state carried
throughout. Reports carried-state ppl per window range; a random-trained model with unexplored states should drift UP
late; a stateful-trained model should stay flat. Eval only. Usage: python temp/stream_eval_long.py name=ckpt [...]"""
import sys, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, BIN
S, W, NWIN = 16, 512, 120
RANGES = [(1, 10), (10, 40), (40, 80), (80, 120)]
torch.set_float32_matmul_precision("high")
d = np.memmap(BIN, dtype=np.uint32, mode="r"); lo = int(len(d) * 0.999)
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    lut = np.load(f"/home/glenn/projects/neuro/data/vocab_map_{c['vocab']}.npy")
    toks = lut[np.asarray(d[lo:lo + S * (NWIN * W + 1)], dtype=np.int64)].reshape(S, NWIN * W + 1)
    ce_w = []; carry = None; mnorm = []
    with torch.no_grad():
        for i in range(NWIN):
            x = torch.from_numpy(toks[:, i * W:(i + 1) * W]).cuda(); y = torch.from_numpy(toks[:, i * W + 1:(i + 1) * W + 1]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, carry = m(x, initial_state=carry, return_final_state=True)
            ce_w.append(F.cross_entropy(out.float().reshape(-1, out.size(-1)), y.reshape(-1)).item())
            mnorm.append(max(l["M"].float().norm(dim=(-2, -1)).mean().item() for l in carry["layers"]))
    ce_w = np.array(ce_w)
    print(f"{name:12s} carried ppl by window " + "  ".join(f"{a}-{b}: {math.exp(ce_w[a:b].mean()):6.2f}" for a, b in RANGES)
          + f" | drift last/first-range {100 * (math.exp(ce_w[80:120].mean() - ce_w[1:10].mean()) - 1):+5.1f}%"
          + f" | ||M|| win1 {mnorm[1]:.1f} -> win119 {mnorm[-1]:.1f}", flush=True)
    del m; torch.cuda.empty_cache()
print("note: the 4 ranges cover different text, so compare MODELS within a range; drift = within-model trend")
