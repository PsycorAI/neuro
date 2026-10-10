"""mem_write_probe.py -- 2026-10-06 (Neuro 52, diagnostic E2). What does the memory write, keep and match?
Uses the GDN-2 audit capture (block.audit = True -> k, q, write gate w, decay alpha per step). Two probes:
 (a) MQAR haystack (kv block at the start of chunk A, query at chunk-B offset 0) -- the case that works;
 (b) xcopy (span in chunk A ending `dist` before the boundary, repeated at chunk-B offset 16), real and random spans.
Per probe it reports:
  w      mean write gate on the tokens that must be stored vs the other chunk-A tokens (selectivity)
  surv   mean per-channel decay survival from the stored token to the read (prod alpha; forgetting)
  match  top-1 / top-3 hit rate: is the stored position s* the best-scoring write for the read query, by
         score(s,t) = sum_c q_t,c * kn_s,c * prod_{u in (s,t]} alpha_u,c (key matching; ±1 position tolerance)
  rank   median rank of s* among all earlier writes (1 = best)
s* = the position whose value holds the token to predict (MQAR: the value token; xcopy: span token j+1 for the read at
span token j). This ignores erase/interference inside the delta rule -- it is a key-matching diagnostic, not a replay.
  python temp/mem_write_probe.py [--chunk 512] name=ckpt.pt [...]"""
import sys, argparse, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN, MQAR_KEY_LO, MQAR_KEY_HI, MQAR_VAL_LO, MQAR_VAL_HI
torch.set_float32_matmul_precision("high")
ap = argparse.ArgumentParser(); ap.add_argument("--chunk", type=int, default=0)
ap.add_argument("--reps", type=int, default=2); ap.add_argument("runs", nargs="+")
a = ap.parse_args()
B, S, CUE, DST = 8, 32, 4, 16


def run(m, x):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m(x.cuda())
    return [b._audit for b in m.blocks if getattr(b, "audit", False)]


def score_stats(au, i, t, s_star, lo=None):
    """Key-match score of query t against every earlier write s < t for batch row i."""
    k = F.normalize(au["k"][i].double(), dim=-1)          # (T, D)
    q = au["q"][i].double()
    la = torch.log(au["a"][i].double().clamp_min(1e-30))  # (T, D)
    L = torch.cumsum(la, 0)
    surv = torch.exp((L[t] - L[:t]).clamp(max=0))          # (t, D): prod alpha over (s, t]
    sc = (k[:t] * surv * q[t]).sum(-1)                     # (t,)
    order = torch.argsort(sc, descending=True)
    rank = (order == s_star).nonzero()[0, 0].item() + 1
    top3 = order[:3].tolist()
    hit1 = abs(order[0].item() - s_star) <= 1
    hit3 = any(abs(p - s_star) <= 1 for p in top3)
    return hit1, hit3, rank, surv[s_star].mean().item()


def report(tag, ws, wo, sv, h1, h3, rk):
    print(f"   {tag:22s} w {np.mean(ws):.3f} vs other {np.mean(wo):.3f} | surv {np.mean(sv):.3f} | "
          f"match top1 {100 * np.mean(h1):5.1f}% top3 {100 * np.mean(h3):5.1f}% | median rank {np.median(rk):.0f}",
          flush=True)


for arg in a.runs:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    C = a.chunk or int(c.get("attn_chunk", 0) or 0) or 512
    mem_ids = []
    for li, b in enumerate(m.blocks):
        if hasattr(b, "qkv"):
            b.attn_chunk = C
        elif hasattr(b, "W_w"):
            b.audit = True; mem_ids.append(li)
    print(f"{name:26s} C={C} E2 memory blocks at {mem_ids} (one line set per block)", flush=True)
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                      val_tokens=c.get("data_val_tokens"))
    # (a) MQAR haystack, N8 and N32
    for N in (8, 32):
        acc = {j: ([], [], [], [], [], []) for j in range(len(mem_ids))}
        gen = torch.Generator().manual_seed(7000 + N); np.random.seed(7000 + N)
        for _ in range(a.reps):
            keys = torch.stack([torch.randperm(MQAR_KEY_HI - MQAR_KEY_LO, generator=gen)[:N] for _ in range(B)]) + MQAR_KEY_LO
            vals = torch.stack([torch.randperm(MQAR_VAL_HI - MQAR_VAL_LO, generator=gen)[:N] for _ in range(B)]) + MQAR_VAL_LO
            kv = torch.stack([keys, vals], 2).reshape(B, 2 * N)
            gap, _ = data.batch(B, C - 2 * N, "val", "cpu")
            qi = torch.randint(0, N, (B,), generator=gen)
            x = torch.cat([kv, gap, keys[torch.arange(B), qi].unsqueeze(1)], 1)
            t = x.shape[1] - 1
            for j, au in enumerate(run(m, x)):
                w = au["w"].float().mean(-1).cpu()                 # (B, T)
                for i in range(B):
                    h1, h3, rk, sv = score_stats(au, i, t, 2 * qi[i].item() + 1)
                    for lst, v in zip(acc[j], (w[i, :2 * N].mean().item(), w[i, 2 * N:C].mean().item(), sv, h1, h3, rk)):
                        lst.append(v)
        for j, li in enumerate(mem_ids):
            report(f"blk{li} MQAR N{N}", *acc[j])
    # (b) xcopy, real and random spans, by distance to the boundary
    for content in ("real", "random"):
        for dist in (0, 64, 256, 448):
            acc = {j: ([], [], [], [], [], []) for j in range(len(mem_ids))}
            np.random.seed(11); gen = torch.Generator().manual_seed(5)
            for _ in range(a.reps):
                x, _ = data.batch(B, 2 * C, "val", "cpu")
                src, dst = C - dist - S, C + DST
                if content == "random":
                    x[:, src:src + S] = torch.randint(1000, c["vocab"] - 1000, (B, S), generator=gen)
                x[:, dst:dst + S] = x[:, src:src + S].clone()
                for j, au in enumerate(run(m, x)):
                    w = au["w"].float().mean(-1).cpu()
                    other = torch.ones(2 * C, dtype=torch.bool); other[src:src + S] = False; other[C:] = False
                    for i in range(B):
                        for jj in range(CUE - 1, S - 1, 4):          # read at span token jj -> needs token jj+1
                            h1, h3, rk, sv = score_stats(au, i, dst + jj, src + jj + 1)
                            for lst, v in zip(acc[j], (w[i, src:src + S].mean().item(), w[i, other].mean().item(),
                                                       sv, h1, h3, rk)):
                                lst.append(v)
            for j, li in enumerate(mem_ids):
                report(f"blk{li} xcopy {content} d{dist}", *acc[j])
    del m; torch.cuda.empty_cache()
