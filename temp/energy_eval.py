"""energy_eval.py -- 2026-10-06 (energy plan). Honest per-token op / energy estimate on real val text, vs the SAME model
computed densely (= a same-size transformer-style dense cost). Convention of the SNN literature (45 nm: MAC 4.6 pJ,
AC 0.9 pJ); a neuromorphic-hardware estimate, NOT GPU energy.
Per nn.Linear (forward pre-hook on real inputs):
  - input from a SpikeCode (spike_code): ACs = mean spike count per input x in x out   (zeros cost nothing)
  - integer-valued input (LIF spikes):    ACs = mean |x| x in x out
  - otherwise:                            MACs = nonzero fraction x in x out          (sparse relu2 down-proj counts)
Attention core per layer: QK^T over the visible context (dense MACs) + AV over min(top-k, context). Memory blocks:
3 d^2 MACs per token (state update + read). Dense reference: every Linear dense MACs + the same attention/memory.
  python temp/energy_eval.py name=ckpt.pt [...]"""
import sys, torch, numpy as np
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN
from spike_code import SpikeCode
E_MAC, E_AC = 4.6e-12, 0.9e-12
B, ITERS = 4, 4

for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    T = int(c["block_size"])
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                      val_tokens=c.get("data_val_tokens"))
    coded = {}                                   # data_ptr of a SpikeCode output -> mean spike count
    acc = {}                                     # linear name -> [mac, ac, dense] summed over tokens

    def sc_hook(mod, inp, out):
        coded[out.data_ptr()] = float(mod.last_count)

    def lin_hook(name):
        def f(mod, inp):
            x = inp[0].detach()
            n_tok = x.numel() // x.shape[-1]
            dense = n_tok * mod.in_features * mod.out_features
            a = acc.setdefault(name, [0.0, 0.0, 0.0])
            a[2] += dense
            if x.data_ptr() in coded:
                a[1] += coded[x.data_ptr()] * dense
            elif bool((x == x.round()).all()):
                a[1] += float(x.abs().float().mean()) * dense
            else:
                a[0] += float((x != 0).float().mean()) * dense
        return f
    hooks = [mm.register_forward_hook(sc_hook) for mm in m.modules() if isinstance(mm, SpikeCode)]
    hooks += [mm.register_forward_pre_hook(lin_hook(n)) for n, mm in m.named_modules() if isinstance(mm, torch.nn.Linear)]
    np.random.seed(0); ntok = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(ITERS):
            x, _ = data.batch(B, T, "val", "cuda")
            coded.clear(); m(x); ntok += x.numel()
    for h in hooks:
        h.remove()
    # attention core + memory (per token)
    d = int(c["d_mem"]); C = int(c.get("attn_chunk", 0) or 0) or T
    ctx = (min(C, T) + 1) / 2
    n_attn = sum(1 for b in m.blocks if hasattr(b, "qkv"))
    n_mem = len(m.blocks) - n_attn
    topk = int(c.get("attn_topk", 0) or 0)
    av_n = min(topk, ctx) if topk else ctx
    attn, attn_ac = 0.0, 0.0                      # spike-coded q / v (spike_code "q"/"v") -> accumulates
    for b in m.blocks:
        if not hasattr(b, "qkv"):
            continue
        sc = getattr(b, "sc", {})
        if "q" in sc:
            attn_ac += float(sc["q"].last_count) * d * ctx
        else:
            attn += d * ctx
        if "v" in sc:
            attn_ac += float(sc["v"].last_count) * d * av_n
        else:
            attn += d * av_n
    attn_dense = n_attn * 2 * d * ctx
    mem = n_mem * 3 * d * d
    mac = sum(a[0] for a in acc.values()) / ntok + attn + mem
    ac = sum(a[1] for a in acc.values()) / ntok + attn_ac
    dense = sum(a[2] for a in acc.values()) / ntok + attn_dense + mem
    e_ev, e_dn = mac * E_MAC + ac * E_AC, dense * E_MAC
    groups = {}
    for n, a in acc.items():
        g = n.split(".")[-1] if not n.startswith("blocks") else n.split(".", 2)[-1].replace("ffn.0", "up").replace("ffn.2", "down")
        gg = groups.setdefault(g, [0.0, 0.0, 0.0])
        for i in range(3):
            gg[i] += a[i] / ntok
    top = sorted(groups.items(), key=lambda kv: -(kv[1][0] * E_MAC + kv[1][1] * E_AC))[:6]
    print(f"{name:28s} energy {e_ev * 1e6:6.1f} uJ/tok vs dense {e_dn * 1e6:6.1f} -> {e_dn / e_ev:4.2f}x | "
          f"MAC {mac / 1e6:5.1f}M AC {ac / 1e6:5.1f}M (dense {dense / 1e6:5.1f}M) | top: " +
          ", ".join(f"{g} {(v[0] * E_MAC + v[1] * E_AC) * 1e6:.1f}uJ" for g, v in top), flush=True)
    del m; torch.cuda.empty_cache()
