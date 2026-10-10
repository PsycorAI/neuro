"""multichunk_eval.py -- 2026-10-06 (Neuro 52, diagnostic E3). Does the memory keep a fact across MANY chunks?
Training sees one chunk boundary (T1024 = 2 chunks); a demo chat crosses many.
 (a) MQAR haystack: kv block at the start of chunk 1, real val text after it, query at offset 0 of chunk k
     (k = 2, 3, 4, 8 -> T up to 8C). Among-candidates accuracy; memory-ablated at k = 2 as the chance check.
 (b) per-chunk val ppl over T = 8C with the memory on / ablated (does the stream degrade as chunks pile up?).
  python temp/multichunk_eval.py [--chunk 512] name=ckpt.pt [...]"""
import sys, math, argparse, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN, MQAR_KEY_LO, MQAR_KEY_HI, MQAR_VAL_LO, MQAR_VAL_HI
torch.set_float32_matmul_precision("high")
ap = argparse.ArgumentParser(); ap.add_argument("--chunk", type=int, default=0)
ap.add_argument("--reps", type=int, default=4); ap.add_argument("--B", type=int, default=8)
ap.add_argument("runs", nargs="+")
a = ap.parse_args()
KS, NCH = (2, 3, 4, 8), 8
B = a.B


def hay(N, k, C, data, gen):
    keys = torch.stack([torch.randperm(MQAR_KEY_HI - MQAR_KEY_LO, generator=gen)[:N] for _ in range(B)]) + MQAR_KEY_LO
    vals = torch.stack([torch.randperm(MQAR_VAL_HI - MQAR_VAL_LO, generator=gen)[:N] for _ in range(B)]) + MQAR_VAL_LO
    kv = torch.stack([keys, vals], 2).reshape(B, 2 * N)
    gap, _ = data.batch(B, (k - 1) * C - 2 * N, "val", "cpu")
    clash = (gap.unsqueeze(-1) == keys.unsqueeze(1)).any(-1)
    gap = torch.where(clash, torch.randint(6000, 8000, gap.shape, generator=gen), gap)
    qi = torch.randint(0, N, (B,), generator=gen)
    return torch.cat([kv, gap, keys[torch.arange(B), qi].unsqueeze(1)], 1), vals, qi


for arg in a.runs:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    C = a.chunk or int(c.get("attn_chunk", 0) or 0) or 512
    for b in m.blocks:
        if hasattr(b, "qkv"):
            b.attn_chunk = C
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                      val_tokens=c.get("data_val_tokens"), prelut=c.get("data_prelut", False))
    for N in (8, 32):
        cols = []
        for k in KS + ("abl",):
            kk = 2 if k == "abl" else k
            gen = torch.Generator().manual_seed(9000 + 10 * N + kk); np.random.seed(9000 + 10 * N + kk)
            hit = tot = 0
            for _ in range(a.reps):
                x, vals, qi = hay(N, kk, C, data, gen)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = m(x.cuda(), ablate_memory=(k == "abl"))[:, -1, :].float().cpu()
                hit += (lg.gather(1, vals).argmax(1) == qi).sum().item(); tot += B
            cols.append(f"{'off@2' if k == 'abl' else f'chunk{k}'} {100.0 * hit / tot:5.1f}")
        print(f"{name:26s} C={C} E3 haystack N{N} (chance {100 / N:.1f}) | " + " | ".join(cols), flush=True)
    T = NCH * C
    res = {}
    for abl in (False, True):
        np.random.seed(0)
        nll = torch.zeros(T, dtype=torch.float64); n = 0
        for _ in range(a.reps * 2):
            x, y = data.batch(B // 2, T, "val", "cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(x, ablate_memory=abl)
            nll += F.cross_entropy(lg.float().reshape(-1, c["vocab"]), y.reshape(-1), reduction="none"
                                   ).reshape(B // 2, T).sum(0).double().cpu()
            n += B // 2
        nll /= n
        res[abl] = [math.exp(nll[i * C:(i + 1) * C].mean().item()) for i in range(NCH)]
    print(f"{name:26s} C={C} E3 ppl by chunk (T={T}) mem on  | " + " ".join(f"{v:6.2f}" for v in res[False]), flush=True)
    print(f"{name:26s} C={C} E3 ppl by chunk (T={T}) mem off | " + " ".join(f"{v:6.2f}" for v in res[True]), flush=True)
    del m; torch.cuda.empty_cache()
