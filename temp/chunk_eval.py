"""chunk_eval.py -- 2026-10-04 (r25). Per-chunk val perplexity with the memory on vs off, under chunked attention.
Each val window of T tokens is split into aligned attention chunks of C tokens (attention never crosses a boundary), so
chunk 1 is the plain in-window case and chunks 2.. can reach earlier chunks ONLY through the memory layer(s).
Reports, per model: ppl of chunk 1 | chunks 2.. | the first 64 tokens of chunks 2.. (where the memory matters most),
each with memory on / ablated. --chunk overrides the model's attn_chunk (needed for models trained without chunks).
  python temp/chunk_eval.py [--chunk 512] [--T 1024] [--iters 25] [--B 8] name=ckpt.pt [name=ckpt.pt ...]"""
import sys, math, argparse, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN
torch.set_float32_matmul_precision("high")
ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, default=0)
ap.add_argument("--T", type=int, default=1024)
ap.add_argument("--iters", type=int, default=25)
ap.add_argument("--B", type=int, default=8)
ap.add_argument("runs", nargs="+")
a = ap.parse_args()
HEAD = 64


def load(path):
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).to("cuda"); m.load_state_dict(s["model"], strict=False); m.eval()
    return m, c


for arg in a.runs:
    name, path = arg.split("=", 1)
    m, c = load(path)
    C = a.chunk or int(c.get("attn_chunk", 0) or 0)
    assert C and a.T % C == 0 and a.T > C, f"{name}: need a chunk size (model attn_chunk or --chunk) dividing T > C"
    for b in m.blocks:
        if hasattr(b, "qkv"):
            b.attn_chunk = C
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                      val_tokens=c.get("data_val_tokens"), prelut=c.get("data_prelut", False))
    pos = torch.arange(a.T)
    first, later, head, head1 = pos < C, pos >= C, (pos >= C) & (pos % C < HEAD), pos < HEAD
    res = {}
    for abl in (False, True):
        np.random.seed(0)
        nll = torch.zeros(a.T, dtype=torch.float64); n = 0
        for _ in range(a.iters):
            x, y = data.batch(a.B, a.T, "val", "cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(x, ablate_memory=abl)
            nll += F.cross_entropy(lg.float().reshape(-1, c["vocab"]), y.reshape(-1), reduction="none"
                                   ).reshape(a.B, a.T).sum(0).double().cpu()
            n += a.B
        nll /= n
        res[abl] = [math.exp(nll[mask].mean().item()) for mask in (first, later, head, head1)]
    on, off = res[False], res[True]
    print(f"{name:26s} C={C} T={a.T} | chunk1 {on[0]:7.2f} (mem off {off[0]:7.2f}) | chunks2+ {on[1]:7.2f} "
          f"(off {off[1]:7.2f}, Δ {off[1] - on[1]:+.2f}) | first{HEAD} of 2+ {on[2]:7.2f} (off {off[2]:7.2f}, "
          f"Δ {off[2] - on[2]:+.2f}) vs first{HEAD} of chunk1 {on[3]:7.2f}", flush=True)
    del m; torch.cuda.empty_cache()
