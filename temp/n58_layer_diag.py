"""n58_layer_diag.py -- 2026-10-12 (Neuro 58). Why are the attention FFNs of blocks 0-5 silent in the 1B run?
Per block on real val text: RMS of the block's input and output stream; for the FFN activation (SparseAct, relu2):
share of positive pre-activations, pre-activation mean / std, output RMS. Plus the weight norms of every parameter
whose name contains "emb" and the token-embedding row norms (seen vs unseen tokens in the batch).
Eval only. Usage: CUDA_VISIBLE_DEVICES=1 python temp/n58_layer_diag.py [--B 2] name=ckpt [...]"""
import sys, argparse, numpy as np, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN
from hybrid_attn import SparseAct
ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2); ap.add_argument("runs", nargs="+")
a = ap.parse_args()
rms = lambda t: t.detach().float().pow(2).mean().sqrt().item()
first = lambda o: o[0] if isinstance(o, (tuple, list)) else o
for arg in a.runs:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    st = {}
    for i, b in enumerate(m.blocks):
        def bh(mod, inp, out, _i=i):
            x = first(inp[0]) if inp else None
            st.setdefault(_i, {}).update(inp=rms(x) if torch.is_tensor(x) else float("nan"), out=rms(first(out)))
        b.register_forward_hook(bh)
        for mod in b.modules():
            if isinstance(mod, SparseAct):
                def ah(mod_, inp, out, _i=i):
                    p = inp[0].detach().float()
                    st.setdefault(_i, {}).update(pos=(p > 0).float().mean().item(), mean=p.mean().item(),
                                                 std=p.std().item(), act=rms(out))
                mod.register_forward_hook(ah)
    T = int(c.get("block_size", 1024))
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                      val_tokens=c.get("data_val_tokens"), prelut=c.get("data_prelut", False))
    np.random.seed(0); x, _ = data.batch(a.B, T, "val", "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m(x)
    print(f"{name}: per block (stream RMS in -> out | FFN pre-activation: positive share, mean, std | FFN activation RMS)")
    for i in sorted(st):
        d = st[i]
        ffn = (f" | pos {d['pos']:.4f} mean {d['mean']:+.3f} std {d['std']:.3f} | act RMS {d['act']:.4f}" if "pos" in d
               else " | memory block")
        print(f"  blk{i:2d} stream {d.get('inp', float('nan')):.3f} -> {d.get('out', float('nan')):.3f}{ffn}", flush=True)
    for n, p in m.named_parameters():
        if "emb" in n.lower():
            line = f"  {n} {tuple(p.shape)} norm {p.detach().float().norm():.1f} rms {rms(p):.4f}"
            if p.dim() == 2 and p.shape[0] == c["vocab"]:
                rn = p.detach().float().norm(dim=1); seen = torch.zeros(c["vocab"], dtype=torch.bool, device=rn.device)
                seen[x.reshape(-1)] = True
                line += (f" | row norm: seen tokens median {rn[seen].median():.3f}, unseen median {rn[~seen].median():.3f}, "
                         f"min {rn.min():.4f}, max {rn.max():.2f}")
            print(line, flush=True)
    del m; torch.cuda.empty_cache()
