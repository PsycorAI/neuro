"""n58_spike_diag.py -- 2026-10-12 (Neuro 58). What does a 0.1-0.3% spike rate in the v4 memory layer look like?
On real val text (T = block_size, so attention chunk 2 is included), for each spiking block:
  rate            mean |s| (the number the trainer prints) and the share of (neuron, token) pairs that fire
  per token       active neurons per token: mean / median / 5th / 95th percentile, share of tokens with NONE active
  per neuron      never fires (dead), fires on > 50% of tokens, how many neurons carry 90% of all spikes
  code            overlap (Jaccard) of the active sets of two positions: same token id vs different token id
                  -> is the code a usable address (distinct per token, repeatable for the same token)?
  signal          RMS of the block's output vs RMS of its input (how loud the memory layer is in the stream)
  by chunk        fire share in attention chunk 1 vs chunk 2..
Eval only. Usage: CUDA_VISIBLE_DEVICES=0 python temp/n58_spike_diag.py [--B 4] name=ckpt [...]"""
import sys, argparse, numpy as np, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN
ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=4)
ap.add_argument("runs", nargs="+")
a = ap.parse_args()
for arg in a.runs:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    cap = []
    for bi, b in enumerate(m.blocks):
        if hasattr(b, "qkv"): continue
        orig = b.forward_sequence
        def fs(*args, _o=orig, _bi=bi, **k):
            out = _o(*args, **k)
            cap.append((_bi, args[0].detach().float(), out[0].detach().float(), out[3].detach().float()))
            return out
        b.forward_sequence = fs
    T = int(c.get("block_size", 1024)); C = int(c.get("attn_chunk", 0) or T)
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                      val_tokens=c.get("data_val_tokens"), prelut=c.get("data_prelut", False))
    np.random.seed(0); x, _ = data.batch(a.B, T, "val", "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m(x)
    for bi, xin, xout, sp in cap:
        N = sp.shape[-1]; ab = sp.abs(); nz = ab > 0
        per_tok = nz.sum(-1).float().flatten()                     # active neurons per token
        per_neu = nz.float().mean((0, 1))                          # share of tokens each neuron fires on
        q = torch.quantile(per_tok, torch.tensor([0.05, 0.5, 0.95], device=per_tok.device))
        srt = torch.sort(per_neu, descending=True).values; cum = torch.cumsum(srt, 0) / srt.sum().clamp_min(1e-9)
        n90 = int((cum < 0.9).sum().item()) + 1
        pos = (sp > 0).float().sum() / nz.float().sum().clamp_min(1)
        print(f"{name} blk{bi} N={N} T={T} chunk={C}", flush=True)
        print(f"  rate      mean|s| {ab.mean():.4f} | fire share {nz.float().mean():.4f} | positive share of spikes {pos:.2f}")
        print(f"  per token active neurons mean {per_tok.mean():.1f} | median {q[1]:.0f} | 5% {q[0]:.0f} | 95% {q[2]:.0f} "
              f"| tokens with none {(per_tok == 0).float().mean():.1%} | with fewer than 3 {(per_tok < 3).float().mean():.1%}")
        print(f"  per neuron dead {(per_neu == 0).float().mean():.1%} | fires on >50% of tokens {(per_neu > 0.5).float().mean():.2%} "
              f"| neurons carrying 90% of spikes {n90} of {N} ({n90 / N:.1%})")
        flat = nz.reshape(-1, N); tok = x.reshape(-1)
        g = torch.Generator(device="cpu").manual_seed(0); idx = torch.randperm(flat.shape[0], generator=g)[:2000].cuda()
        A = flat[idx].float(); cnt = A.sum(1); inter = A @ A.T
        jac = inter / (cnt[:, None] + cnt[None] - inter).clamp_min(1)
        same = tok[idx][:, None] == tok[idx][None]; off = ~torch.eye(len(idx), dtype=torch.bool, device=same.device)
        both = (cnt[:, None] > 0) & (cnt[None] > 0) & off
        print(f"  code      overlap, same token id {jac[same & both].mean():.3f} (n={int((same & both).sum())}) "
              f"| different token id {jac[~same & both].mean():.3f}")
        print(f"  signal    output RMS {xout.pow(2).mean().sqrt():.3f} | input RMS {xin.pow(2).mean().sqrt():.3f}")
        if T > C:
            print(f"  by chunk  fire share chunk 1 {nz[:, :C].float().mean():.4f} | chunks 2.. {nz[:, C:].float().mean():.4f}")
    del m, cap; torch.cuda.empty_cache()
