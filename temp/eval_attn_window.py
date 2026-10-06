"""eval_attn_window.py -- 2026-09-30: can a model run with a SHORT attention window at inference (constant memory at any
context length)? Standard val windows (EVAL_SEED, 512 tok, zero state) with the attention layers' attn_window forced to
32 / 64 / 128 / full. Compares r9b (trained with a stochastic 64/full window) vs the signed controls. Eval only."""
import sys, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN, EVAL_SEED
torch.set_float32_matmul_precision("high")
WINS, ITERS = (32, 64, 128, 0), 40
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256))
    attn = [b for b in m.blocks if hasattr(b, "qkv")]
    res = []
    for w in WINS:
        for b in attn: b.attn_window = w
        np.random.seed(EVAL_SEED); tot = 0.0
        for _ in range(ITERS):
            x, y = data.batch(8, 512, "val", "cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tot += F.cross_entropy(m(x).float().reshape(-1, c["vocab"]), y.reshape(-1)).item()
        res.append(math.exp(tot / ITERS))
    print(f"{name:10s} val_ppl at attn window 32/64/128/full: " + " / ".join(f"{p:6.2f}" for p in res)
          + f" | w64 vs full {100 * (res[1] / res[3] - 1):+5.1f}%", flush=True)
    del m; torch.cuda.empty_cache()
