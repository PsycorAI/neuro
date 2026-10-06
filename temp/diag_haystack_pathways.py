"""diag_haystack_pathways.py -- 2026-09-29: WHY does gapped recall fail in the hybrid / lev4 stacks?
For each ckpt, haystack N8 accuracy (same batch recipe as temp/haystack_recall_eval.py) under:
  full      -- unmodified
  mem_off   -- GDN memory read ablated (ablate_memory=True)
  attn_off  -- attention mixing zeroed in attention blocks (FFN kept); hybrids only
plus the GDN decay RETENTION of the kv block across the gap: prod of alpha over positions after the kv block,
per (sample, channel); reported as mean over channels (the read sums channels) and the median per-channel half-life
on the gap text (tokens). Eval only, small (B64 x 4 reps).
  CUDA_VISIBLE_DEVICES=1 python temp/diag_haystack_pathways.py name=ckpt [name=ckpt ...]"""
import sys, math, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import model as MM
from train_gpu import build_model, StreamText, BIN, MQAR_KEY_LO, MQAR_KEY_HI, MQAR_VAL_LO, MQAR_VAL_HI
import numpy as np
N, GAPS, B, REPS = 8, (0, 64, 192), 64, 4
torch.set_float32_matmul_precision("high")

_alphas = []
_orig_alpha_stats = MM._alpha_stats
def _stash(alpha_seq, **kw):
    _alphas.append(alpha_seq.detach().float())
    return _orig_alpha_stats(alpha_seq, **kw)
MM._alpha_stats = _stash

def batch(g, data, gen):
    keys = torch.stack([torch.randperm(MQAR_KEY_HI - MQAR_KEY_LO, generator=gen)[:N] for _ in range(B)]) + MQAR_KEY_LO
    vals = torch.stack([torch.randperm(MQAR_VAL_HI - MQAR_VAL_LO, generator=gen)[:N] for _ in range(B)]) + MQAR_VAL_LO
    parts = [torch.stack([keys, vals], 2).reshape(B, 2 * N)]
    if g > 0:
        gap, _ = data.batch(B, g, "val", "cpu")
        clash = (gap.unsqueeze(-1) == keys.unsqueeze(1)).any(-1)
        parts.append(torch.where(clash, torch.randint(6000, 8000, gap.shape, generator=gen), gap))
    qi = torch.randint(0, N, (B,), generator=gen)
    parts.append(keys[torch.arange(B), qi].unsqueeze(1))
    return torch.cat(parts, 1), vals, qi

def run(m, data, mode):
    attn_blocks = [b for b in m.blocks if hasattr(b, "qkv")]
    if mode == "attn_off":
        if not attn_blocks: return None
        for b in attn_blocks: b._attn = lambda x: torch.zeros_like(x)
    acc, ret, hl = {}, {}, {}
    for g in GAPS:
        gen = torch.Generator().manual_seed(1000 * N + g); np.random.seed(1000 * N + g)
        hit = tot = 0; r_list = []; h_list = []
        for _ in range(REPS):
            x, vals, qi = batch(g, data, gen)
            _alphas.clear()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(x.cuda(), ablate_memory=(mode == "mem_off"))[:, -1, :].float().cpu()
            hit += (lg.gather(1, vals).argmax(1) == qi).sum().item(); tot += B
            if mode == "full" and _alphas:
                for a in _alphas:                                   # one per spiking block, (B, T, dk)
                    la = torch.log(a[:, 2 * N:, :].clamp_min(1e-12))  # everything after the kv block (gap + query)
                    r_list.append(la.sum(1).exp().mean().item())      # mean over (sample, channel) of retention
                    if g > 0:
                        per_ch = la[:, :-1, :].mean((0, 1))           # mean log-alpha per channel on gap text
                        h_list.append(float(torch.median(math.log(0.5) / per_ch.clamp(max=-1e-9))))
        acc[g] = 100.0 * hit / tot
        if r_list: ret[g] = sum(r_list) / len(r_list)
        if h_list: hl[g] = sum(h_list) / len(h_list)
    for b in attn_blocks: b.__dict__.pop("_attn", None)
    return acc, ret, hl

for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).to("cuda"); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256))
    for mode in ("full", "mem_off", "attn_off"):
        r = run(m, data, mode)
        if r is None: continue
        acc, ret, hl = r
        line = f"{name:12s} {mode:8s} N8 acc gap0/64/192: " + " ".join(f"{acc[g]:5.1f}" for g in GAPS)
        if ret: line += " | kv retention: " + " ".join(f"{ret[g]:.3f}" for g in GAPS)
        if hl: line += " | median half-life on text (tok): " + " ".join(f"{hl[g]:.0f}" for g in GAPS if g in hl)
        print(line, flush=True)
    del m; torch.cuda.empty_cache()
print(f"chance = {100 / N:.1f}%; retention = mean over channels of prod(alpha) from after the kv block to the query")
