"""energy_eval_v2.py -- 2026-10-06 (Neuro 53, diagnostic E4: honest energy with data movement).
energy_eval.py counts only arithmetic (45 nm MAC 4.6 pJ / AC 0.9 pJ). Real chips also pay to READ each weight, and that
usually dominates. This version adds a weight-fetch term per Linear:
  dense:   every weight is read once per token                        -> in x out reads
  spiking: only the weights of NONZERO inputs are read (a spike count s needs one read, then |s| adds)
           -> nonzero fraction x in x out reads
Two memory scenarios (Horowitz ISSCC 2014, 45 nm, scaled to 16-bit weights):
  SRAM on chip (weights spread over cores, neuromorphic style) ~2.5 pJ / read  (8 KB SRAM: 10 pJ per 64 bit)
  DRAM streamed (batch-1 decoding of a model too big for SRAM)  ~320 pJ / read (DRAM: ~1.3-2.6 nJ per 64 bit)
Attention KV reads (2 x d x context per layer) and the memory state (3 d^2) are charged as dense SRAM reads in BOTH
models (conservative for the spiking model). Prints compute-only (= energy_eval.py), +SRAM, +DRAM ratios vs dense.
  python temp/energy_eval_v2.py name=ckpt.pt [...]"""
import sys, torch, numpy as np
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN
from spike_code import SpikeCode
E_MAC, E_AC, E_SRAM, E_DRAM = 4.6e-12, 0.9e-12, 2.5e-12, 320e-12
B, ITERS = 4, 4

for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    T = int(c["block_size"])
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                      val_tokens=c.get("data_val_tokens"), prelut=c.get("data_prelut", False))
    coded = {}
    acc = {}                                     # linear name -> [mac, ac, dense, reads]

    def sc_hook(mod, inp, out):
        coded[out.data_ptr()] = float(mod.last_count)

    def lin_hook(name):
        def f(mod, inp):
            x = inp[0].detach()
            n_tok = x.numel() // x.shape[-1]
            dense = n_tok * mod.in_features * mod.out_features
            nz = float((x != 0).float().mean())
            a = acc.setdefault(name, [0.0, 0.0, 0.0, 0.0])
            a[2] += dense
            a[3] += nz * dense
            if x.data_ptr() in coded:
                a[1] += coded[x.data_ptr()] * dense
            elif bool((x == x.round()).all()):
                a[1] += float(x.abs().float().mean()) * dense
            else:
                a[0] += nz * dense
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
    d = int(c["d_mem"]); C = int(c.get("attn_chunk", 0) or 0) or T
    ctx = (min(C, T) + 1) / 2
    n_attn = sum(1 for b in m.blocks if hasattr(b, "qkv"))
    n_mem = len(m.blocks) - n_attn
    topk = int(c.get("attn_topk", 0) or 0)
    av_n = min(topk, ctx) if topk else ctx
    attn, attn_ac = 0.0, 0.0
    for b in m.blocks:
        if not hasattr(b, "qkv"):
            continue
        sc = getattr(b, "sc", {})
        attn_ac += float(sc["q"].last_count) * d * ctx if "q" in sc else 0.0
        attn += 0.0 if "q" in sc else d * ctx
        attn_ac += float(sc["v"].last_count) * d * av_n if "v" in sc else 0.0
        attn += 0.0 if "v" in sc else d * av_n
    attn_dense = n_attn * 2 * d * ctx
    mem = n_mem * 3 * d * d
    side_reads = n_attn * 2 * d * ctx + n_mem * 3 * d * d          # KV cache + memory state, same in both
    mac = sum(a[0] for a in acc.values()) / ntok + attn + mem
    ac = sum(a[1] for a in acc.values()) / ntok + attn_ac
    dense = sum(a[2] for a in acc.values()) / ntok + attn_dense + mem
    rd_sp = sum(a[3] for a in acc.values()) / ntok
    rd_dn = sum(a[2] for a in acc.values()) / ntok
    comp_sp, comp_dn = mac * E_MAC + ac * E_AC, dense * E_MAC
    side = side_reads * E_SRAM
    out = [f"compute only {comp_dn / comp_sp:4.2f}x ({comp_sp * 1e6:.0f} vs {comp_dn * 1e6:.0f} uJ)"]
    for tag, e in (("+SRAM fetch", E_SRAM), ("+DRAM fetch", E_DRAM)):
        sp, dn = comp_sp + rd_sp * e + side, comp_dn + rd_dn * e + side
        out.append(f"{tag} {dn / sp:4.2f}x ({sp * 1e6:.0f} vs {dn * 1e6:.0f} uJ)")
    print(f"{name:28s} energy-v2 | weight reads {rd_sp / 1e6:.1f}M of {rd_dn / 1e6:.1f}M ({100 * rd_sp / rd_dn:.0f}%) | "
          + " | ".join(out), flush=True)
    del m; torch.cuda.empty_cache()
