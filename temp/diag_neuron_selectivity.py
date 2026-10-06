"""diag_neuron_selectivity.py -- 2026-10-01 (interpretability first look, CPU only, eval only).
Question: are our spiking neurons more selective ("closer to one feature per unit") than dense SwiGLU MLP units?
For every unit we split signs into two half-units (+ and -). A half-unit's ACTIVE SET is:
  spiking neuron: the tokens where it fires with that sign;
  SwiGLU hidden unit: its top-p tokens by signed value, p = the spiking layer's mean firing rate per sign
  (matched activity density, so sparsity itself cannot drive the score).
Scores over each active set (higher = more selective):
  top1 = share of the active set taken by its single most common current token
  top5 = same for its 5 most common tokens
  ent  = 1 - H(current token | active) / H(current token overall)   (0 = no token preference)
Reported per layer as medians over half-units with >= MIN_ACTIVE tokens. Depth confound: compare layer 0 with layer 0.
CPU only (the LIF Triton kernel is swapped for an exact torch reference). Usage:
  python temp/diag_neuron_selectivity.py name=ckpt [name=ckpt ...]"""
import sys, math, numpy as np, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import lif_exact as LE
from train_gpu import build_model, StreamText, BIN

def lif_ml_ref(cur, beta, vth=1.0, K=None, alpha=2.0, gamma=1.0, levels=1, signed=False):
    B, T, N = cur.shape
    beta = beta if torch.is_tensor(beta) else torch.full((N,), float(beta))
    mem = torch.zeros(B, N); out = torch.empty(B, T, N)
    for t in range(T):
        pre = beta * mem + cur[:, t].float()
        s = torch.clamp(torch.floor(pre / vth), 0, levels)
        if signed:
            s = s - torch.clamp(torch.floor(-pre / vth), 0, levels)
        out[:, t] = s; mem = pre - gamma * s * vth
    return out.to(cur.dtype), mem.to(cur.dtype)
LE.lif_exact_ml = lif_ml_ref
LE.lif_exact = lambda cur, beta, vth=1.0, K=None, alpha=2.0, gamma=1.0: lif_ml_ref(cur, beta, vth, gamma=gamma)

B, T, MIN_ACTIVE = 8, 512, 20
torch.set_num_threads(8)

def scores(act_mask, toks, H_all):
    """act_mask (P, U) bool over token positions; toks (P,) ints. Returns per-unit (n_active, top1, top5, ent)."""
    res = []
    for u in range(act_mask.shape[1]):
        idx = act_mask[:, u].nonzero().squeeze(1)
        n = idx.numel()
        if n < MIN_ACTIVE:
            continue
        _, cnt = torch.unique(toks[idx], return_counts=True)
        p = cnt.float() / n
        srt = torch.sort(p, descending=True).values
        H = -(p * p.log()).sum().item()
        res.append((n, srt[0].item(), srt[:5].sum().item(), 1 - H / H_all))
    return np.array(res) if res else np.zeros((0, 4))

for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    cap = {}
    for i, b in enumerate(m.blocks):
        if hasattr(b, "qkv"):
            (b.ffn.w2 if hasattr(b.ffn, "w2") else b.ffn[-1]).register_forward_hook(lambda mod, inp, out, i=i: cap.__setitem__(i, ("mlp", inp[0].detach().float())))
        else:
            orig = b.forward_sequence
            def fs(*a, _o=orig, _i=i, **k):
                out = _o(*a, **k); cap[_i] = ("spk", out[3].detach().float()); return out
            b.forward_sequence = fs
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256))
    np.random.seed(0); x, _ = data.batch(B, T, "val", "cpu")
    with torch.no_grad():
        m(x)
    toks = x.reshape(-1).cpu()
    _, cnt = torch.unique(toks, return_counts=True); p = cnt.float() / toks.numel()
    H_all = -(p * p.log()).sum().item()
    rate = [((v[1] > 0).float().mean().item(), (v[1] < 0).float().mean().item()) for v in cap.values() if v[0] == "spk"]
    rate = rate[0] if rate else (0.05, 0.05)
    print(f"== {name}: {toks.numel()} tokens, H(token) {H_all:.2f} nats; matched density +{rate[0]:.3f} / -{rate[1]:.3f}")
    for i in sorted(cap):
        kind, a = cap[i]; a = a.reshape(-1, a.shape[-1])
        if kind == "spk":
            masks = [a > 0, a < 0]
        else:
            masks = []
            for sgn, r in ((1, rate[0]), (-1, rate[1])):
                k = max(1, int(round(r * a.shape[0])))
                thr = torch.topk(sgn * a, k, dim=0).values[-1]          # per-unit k-th largest
                masks.append((sgn * a) >= thr)
        sc = np.concatenate([scores(mk, toks, H_all) for mk in masks])
        dead = sum(int((mk.sum(0) < MIN_ACTIVE).sum()) for mk in masks)
        print(f"  L{i} {kind:3s} units {a.shape[1]:5d} | half-units scored {len(sc):5d} (skipped {dead}) | "
              f"median top1 {np.median(sc[:, 1]):.3f}  top5 {np.median(sc[:, 2]):.3f}  ent {np.median(sc[:, 3]):.3f} | "
              f"p90 top1 {np.quantile(sc[:, 1], 0.9):.3f}", flush=True)
    del m
