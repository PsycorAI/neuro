"""head_rank_sens.py -- 2026-10-06: zero-shot val ppl of r25o with the LM head replaced by its rank-r SVD truncation
(same shape; tells whether a factorised d->r->V head, ~V*r ops instead of V*d, is worth building)."""
import sys, math, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN
s = torch.load("/home/glenn/projects/neuro/models/checkpoints/r25o_13L_H6only_150m/step_0012208.pt", map_location="cpu",
               weights_only=False)
c = dict(s["cfg"]); m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval()
data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                  val_tokens=c.get("data_val_tokens"))
W = m.head.weight.data.float().clone()
U, S, Vh = torch.linalg.svd(W, full_matrices=False)


def ppl():
    np.random.seed(0); tot = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(10):
            x, y = data.batch(4, 1024, "val", "cuda")
            tot += F.cross_entropy(m(x).float().reshape(-1, c["vocab"]), y.reshape(-1)).item()
    return math.exp(tot / 10)


out = [f"full {ppl():.2f}"]
for r in (128, 256, 384, 512):
    m.head.weight.data.copy_(((U[:, :r] * S[:r]) @ Vh[:r]).to(m.head.weight.dtype))
    out.append(f"r{r} {ppl():.2f}")
print("head rank:", " | ".join(out), f"| d={W.shape[1]}")
