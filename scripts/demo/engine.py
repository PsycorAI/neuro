"""demo/engine.py -- 2026-10-14 (Neuro 61; docs/demo_v4_plan_2026_10_12.md D2/D3/D5/D6/D7). The model side of the v4 demo,
importable and testable without the HTTP layer (scripts/demo/server.py is the thin web front end).

  * strict load: a checkpoint / config key mismatch is an error (the server refuses to start) unless allow_mismatch.
  * Tracer (RC3): per-position activity for the brain view (memory-layer spikes, FFN fire masks, memory write gate) and
    the energy-v2 meter of temp/energy_eval_v2.py, PER TOKEN: weight reads (dense: in*out; event-driven: nonzero
    inputs * out), compute (spike-coded inputs = accumulates, other nonzero inputs = MACs, dense = MACs), attention
    over the visible context of the token's own aligned chunk (attn_chunk) and the memory state. Known difference from
    energy_eval_v2: for spike-coded attention q / v the eval multiplies the MEAN spike count by the MEAN context, the
    meter uses each token's own count x its own context (exact per token); compute-only differs by ~2% (test: < 3%),
    the weight-fetch scenarios by < 1%. Three scenarios:
    compute only, + SRAM weight fetch (2.5 pJ / read), + DRAM weight fetch (320 pJ / read); attention KV and memory
    state reads charged as SRAM in both models (as energy_eval_v2). 45 nm numbers (Horowitz ISSCC 2014).
  * BrainSlot: ONE brain (LoRA adapter) at a time, added into the weights (delta = alpha/r * B @ A) and restored from
    kept copies; refuses an adapter trained on another base (brains do not transfer between checkpoints: 1B test
    10-14, 95% -> 21%), and never stacks two (two merged brains = 0% / 0%).
  * assistant turn (protocol v2): tools block + history + question -> grammar-constrained decode -> ANSWER text, or
    TOOL <sentence> {json} -> executor (scripts/demo/tools_exec.py; offline switch) -> second pass in the trained
    format `### Instruction / ### Input: Result from <tool>: <text> / ### Response:` -> the reply. History (D5): earlier
    turns as `ANSWER <reply>`, whole prompt cut to `prompt_budget` tokens (the SFT length), oldest turn first.
  * raw-text mode (the 10-04 stage-1 behaviour) for the memory showcase."""
import os, sys, glob, threading
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, HERE)
import torch
import torch.nn as nn
import torch.nn.functional as F

# n76 10-24 (Glenn): DEMO_PRECISION=int8 -> the int8 figures of temp/n74_energy.py (MAC 0.3 = int8 mult 0.2 + int32 add
# 0.1, AC 0.1, 8-bit read SRAM 1.25 / DRAM 160 pJ), applied to the event-driven AND the dense side alike; matches a
# weight-only int8 checkpoint (temp/n76_w8.py, e.g. sft_v4_chat_w8: ppl +0.06%). Default fp = the figures below.
PRECISION = os.environ.get("DEMO_PRECISION", "fp")
E_MAC, E_AC, E_SRAM, E_DRAM = {"fp": (4.6e-12, 0.9e-12, 2.5e-12, 320e-12),
                               "int8": (0.3e-12, 0.1e-12, 1.25e-12, 160e-12)}[PRECISION]
H0_BINS, MLP_BINS = 128, 48
CK = os.path.join(ROOT, "models", "checkpoints")


def patch_cpu_lif():
    """The exact-LIF kernels are Triton (GPU only): torch reference for CPU tests (as temp/parity_r16_flags.py)."""
    import lif_exact as LE

    def _lif_ml_ref(cur, beta, vth=1.0, K=None, alpha=2.0, gamma=1.0, levels=1, signed=False):
        B, T, N = cur.shape
        beta = beta if torch.is_tensor(beta) else torch.full((N,), float(beta))
        mem = torch.zeros(B, N, dtype=cur.dtype)
        outs = []
        for t in range(T):
            pre = beta * mem + cur[:, t]
            hard = torch.clamp(torch.floor(pre / vth), 0, levels)
            if signed:
                hard = hard - torch.clamp(torch.floor(-pre / vth), 0, levels)
            outs.append(hard)
            mem = pre - gamma * hard * vth
        return torch.stack(outs, 1), mem
    LE.lif_exact_ml = _lif_ml_ref
    LE.lif_exact = lambda cur, beta, vth=1.0, K=None, alpha=2.0, gamma=1.0: _lif_ml_ref(cur, beta, vth, gamma=gamma)


def latest(run, pat="step_*.pt"):
    cks = sorted(glob.glob(os.path.join(CK, run, pat)))
    if not cks:
        raise FileNotFoundError(f"no {pat} in models/checkpoints/{run}")
    return cks[-1]


def _v4f_struct():
    """n78 W4: v4f's size from its config (configs/v4std_dense.yaml): weights a token touches (every matrix + the LM
    head; the embedding is one row) and the K + V values it caches. None if the config is not there."""
    import re
    try:
        cfg = open(os.path.join(ROOT, "configs", "v4std_dense.yaml")).read()
        gi = lambda k: int(re.search(rf"^{k}:\s*(\d+)", cfg, re.M).group(1))
        d, L, h, V = gi("d_mem"), gi("n_layers"), gi("attn_ffn_hidden"), gi("vocab")
    except (OSError, AttributeError):
        return None
    return dict(d=d, layers=L, weights_per_token=L * (4 * d * d + 3 * d * h) + V * d, kv_per_token=2 * L * d)


class Tracer:
    """Per-position activity + energy-v2 counts for one batch-1 forward pass."""

    def __init__(self, model, c):
        from hybrid_attn import SparseAct
        from spike_code import SpikeCode
        self.model, self.on = model, False
        self.attn = [b for b in model.blocks if hasattr(b, "qkv")]
        self.spk_blocks = [b for b in model.blocks if not hasattr(b, "qkv")]
        self.d = int(c["d_mem"]); self.C = int(c.get("attn_chunk", 0) or 0); self.topk = int(c.get("attn_topk", 0) or 0)
        self.W = 0                                                  # n78: sliding attention window (engine_stream.py)
        self.v4f = _v4f_struct()                                    # n78 W4: the dense transformer the page compares to
        self.has_q = any("q" in getattr(b, "sc", {}) for b in self.attn)
        self.has_v = any("v" in getattr(b, "sc", {}) for b in self.attn)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.register_forward_pre_hook(self._lin_hook)
            elif isinstance(m, SparseAct):
                m.register_forward_hook(self._act_hook)
            elif isinstance(m, SpikeCode):
                m.register_forward_hook(self._sc_hook)
        for b in self.spk_blocks:
            b.audit = True
            b.telemetry_off = True                                  # n63: no training telemetry (14 host syncs / forward)
            orig = b.forward_sequence

            def wrapped(*a, _orig=orig, **k):
                out = _orig(*a, **k)
                if self.on:
                    self.spikes.append(out[3][0].detach())          # (T, N) spikes of the memory layer
                return out
            b.forward_sequence = wrapped
        self.reset(0)

    def reset(self, T):
        self.T, self.lins, self.acts, self.spikes, self.coded, self.qv = T, [], [], [], {}, []
        self.Tv = None
        for b in self.spk_blocks:
            b._audit = None

    def _sc_hook(self, mod, inp, out):
        """Spike-code output = s * theta: per-position mean |s| (spikes per input element), keyed by the tensor."""
        if not self.on:
            return
        th = mod.theta.detach().abs().to(out.dtype) + 1e-4
        cnt = (out.detach() / th).round().abs()
        if cnt.dim() == 4 and cnt.shape[2] == self.T:                 # (B, H, T, dh): attention q / v codes
            self.qv.append((mod, cnt[0].float().mean(dim=(0, 2))))
        elif cnt.dim() == 3 and cnt.shape[1] == self.T:
            self.coded[out.data_ptr()] = (out, cnt[0].float().mean(-1))   # n63: hold `out` so its address cannot be
                                                                          # reused by a later (uncoded) tensor this pass

    def _lin_hook(self, mod, inp):
        if not self.on:
            return
        x = inp[0]
        if x.dim() != 3 or x.shape[0] != 1 or x.shape[1] != self.T:
            return
        dense = float(mod.in_features * mod.out_features)
        xd = x[0].detach()
        nz = (xd != 0).float().mean(-1)                             # (T,) nonzero share of the input
        e = self.coded.get(x.data_ptr()); cnt = e[1] if e is not None else None; isint = None
        if cnt is None:                                             # integer input (spikes) = additions, as energy_eval_v2;
            isint = (xd == xd.round()).all()                        # n63: kept on the GPU (was a host sync per Linear),
            cnt = xd.abs().float().mean(-1)                         # resolved in per_position's single copy
        self.lins.append((nz, cnt, dense, isint))

    def _act_hook(self, mod, inp, out):
        if self.on and out.dim() == 3 and out.shape[1] == self.T:
            self.acts.append((out[0] != 0).detach())

    def per_position(self, t0):
        T, d = self.T, self.d
        mac = torch.zeros(T); ac = torch.zeros(T); dense = torch.zeros(T); rd_sp = torch.zeros(T); rd_dn = torch.zeros(T)
        if self.lins:                                               # n63: one device->host copy, not one per Linear
            dn_l = torch.tensor([dn for _, _, dn, _ in self.lins])
            nz_l = torch.nan_to_num(torch.stack([nz.float() for nz, _, _, _ in self.lins]).cpu())   # n78: a fully pruned
            # FFN's W_out has 0 inputs (mean of nothing = nan; its dense cost is 0 anyway)
            has = torch.stack([torch.ones((), dtype=torch.bool, device=nz.device) if isint is None else isint
                               for nz, _, _, isint in self.lins]).cpu()
            cnt_l = torch.nan_to_num(torch.stack([cnt.float() for _, cnt, _, _ in self.lins]).cpu())
            dense += dn_l.sum(); rd_dn += dn_l.sum(); rd_sp += (nz_l * dn_l[:, None]).sum(0)
            ac += (cnt_l * dn_l[:, None])[has].sum(0)
            mac += (nz_l * dn_l[:, None])[~has].sum(0)
        pos = torch.arange(T, dtype=torch.float32)
        vis = (pos % self.C) + 1 if self.C else pos + 1               # keys the token's query can see in its chunk
        if self.W:                                                    # n78: sliding window instead of chunks
            vis = (pos + 1).clamp(max=self.W)
        av = vis.clamp(max=self.topk) if self.topk else vis
        qv = {}
        qv_c = torch.stack([cnt for _, cnt in self.qv]).cpu() if self.qv else None   # n63: one copy
        for (mod, _), cnt in zip(self.qv, qv_c if qv_c is not None else []):
            qv.setdefault(id(mod), cnt)
        for b in self.attn:
            sc = getattr(b, "sc", {})
            qc = qv.get(id(sc["q"])) if "q" in sc else None
            vc = qv.get(id(sc["v"])) if "v" in sc else None
            ac += (qc * d * vis) if qc is not None else 0.0
            mac += 0.0 if qc is not None else d * vis
            ac += (vc * d * av) if vc is not None else 0.0
            mac += 0.0 if vc is not None else d * av
            dense += 2 * d * vis
        mem = len(self.spk_blocks) * 3.0 * d * d
        mac += mem; dense += mem
        side = len(self.attn) * 2 * d * vis + mem                      # KV cache + memory state reads (SRAM, both)
        comp_sp, comp_dn = mac * E_MAC + ac * E_AC, dense * E_MAC
        # n78 10-25: the spike / fire-mask reductions run on the GPU over positions t0.. only and just the small results
        # are copied (was: all T positions x 12 x 12288 fire masks as float32 to the CPU, ~160 MB per pass); the
        # inputs are 0/1, so sums, means and bin averages are exact on either device -> the same values
        Tv = self.Tv or T                                             # n78: a right-padded (graph) pass: valid positions
        n = Tv - t0
        spk = [s[t0:Tv].ne(0).float() for s in self.spikes]
        # n78: per block (a pruned copy's FFNs have different widths; a fully pruned one has 0 units = activity 0)
        acts = [x[t0:Tv].float() for x in self.acts]                              # each (n, units)
        writes = [b._audit["w"][0].float().mean(-1).cpu() for b in self.spk_blocks if getattr(b, "_audit", None)]
        n_units = sum(s.shape[1] for s in spk) + sum(x.shape[1] for x in acts)
        # every per-position quantity for all positions at once (same float32 ops element-wise -> the same values; the
        # per-position loop of small tensor ops was slow), then read out as lists
        sl = slice(t0, Tv)
        fired = sum(s.sum(-1) for s in spk) + sum(x.sum(-1) for x in acts)
        fired = fired.cpu().tolist() if torch.is_tensor(fired) else [0] * n
        h0 = F.adaptive_avg_pool1d(spk[0][:, None], H0_BINS)[:, 0].cpu().tolist() if spk else [[] for _ in range(n)]
        dev = acts[0].device if acts else None
        mlp = (torch.stack([x.mean(-1) if x.shape[1] else torch.zeros(n, device=dev) for x in acts]).cpu().tolist()
               if acts else [])
        strips = (torch.stack([F.adaptive_avg_pool1d(x[:, None], MLP_BINS)[:, 0] if x.shape[1]
                               else torch.zeros(n, MLP_BINS, device=dev) for x in acts]).cpu().tolist() if acts else [])
        wr = writes[0][sl].tolist() if writes else None
        en = {}
        for tag, er in (("compute", 0.0), ("sram", E_SRAM), ("dram", E_DRAM)):
            en[tag] = ((comp_sp[sl] + rd_sp[sl] * er + (side[sl] * E_SRAM if er else 0.0)).tolist(),
                       (comp_dn[sl] + rd_dn[sl] * er + (side[sl] * E_SRAM if er else 0.0)).tolist())
        # n78 W4: v4f (the matched dense transformer) on the same hardware model, same figures, same rule (KV / state reads
        # on-chip in both): every weight read and multiplied, attention over ALL earlier words (no window), K + V cached
        en_f = {}
        if self.v4f:
            Wf, kvf = float(self.v4f["weights_per_token"]), float(self.v4f["kv_per_token"])
            att = kvf * (pos[sl] + 1)                                 # QK and AV multiply-adds = values of K + V read
            for tag, er in (("compute", 0.0), ("sram", E_SRAM), ("dram", E_DRAM)):
                en_f[tag] = ((Wf + att) * E_MAC + Wf * er + (att * E_SRAM if er else 0.0)).tolist()
        rs, rd, oe, od = rd_sp[sl].tolist(), rd_dn[sl].tolist(), (mac[sl] + ac[sl]).tolist(), dense[sl].tolist()
        res = []
        for i in range(n):
            e = {tag: dict(pj_event=v[0][i] * 1e12, pj_dense=v[1][i] * 1e12,
                           **({"pj_v4f": en_f[tag][i] * 1e12} if en_f else {})) for tag, v in en.items()}
            res.append(dict(
                h0=[round(v, 3) for v in h0[i]],
                mlp=[round(m_[i], 4) for m_ in mlp],
                mlp_strips=[[round(v, 3) for v in s_[i]] for s_ in strips],
                write=round(wr[i], 3) if writes else None,
                active=round(float(fired[i]) / max(n_units, 1), 4),
                reads_event=rs[i], reads_dense=rd[i], ops_event=oe[i], ops_dense=od[i],
                pj_event=e["compute"]["pj_event"], pj_dense=e["compute"]["pj_dense"], energy=e))
        return res


# 10-20: ordinary text between the word pairs and the question of the memory test (written for it; repeated as needed)
_FILLER = ("The village sat at the edge of a wide valley, where a slow river bent around the fields before it reached "
           "the hills. In spring the farmers walked out early to look at the soil, and in the evening they met at the "
           "small square to talk about the weather, the price of seed and the news from the town across the bridge. "
           "The school had one large room with tall windows, and the children learned to read from old books that "
           "their parents had used before them. On market days the road filled with wagons of apples, bread, cheese "
           "and wool, and the noise could be heard from the mill at the far end of the street. Most people agreed "
           "that the best season was autumn, when the trees turned yellow and red and the air was full of smoke and "
           "rain. Visitors sometimes asked why anyone would stay in such a quiet place, and the answer was always "
           "the same: the work was hard, but the neighbours were kind, the water was clean and the nights were dark "
           "enough to see every star. ")


def _base_tokens(run):
    """10-20: the pretraining tokens of a fine-tuned run's base, from its SFT log ("loaded BASE: .../cF/... (10.51B
    tok)"; a "_slim" copy shares the log of its run). The checkpoint's own count is the fine-tuning tokens only (the
    page said "0.06B training tokens"). 0 when there is no such log (a base model)."""
    import re
    for name in (run, re.sub(r"_slim$", "", run)):
        try:
            hits = re.findall(r"loaded BASE: .*\(([\d.]+)B tok\)", open(os.path.join(ROOT, "logs", name + ".log"),
                                                                        encoding="utf-8", errors="replace").read())
        except OSError:
            continue
        if hits:
            return float(hits[-1]) * 1e9
    return 0


class BrainSlot:
    """One LoRA brain at a time, merged into the weights; the original matrices are kept on the CPU to restore."""

    def __init__(self, model, base_run, brains):
        self.model, self.base_run, self.brains, self.on, self.saved = model, base_run, dict(brains), None, {}
        self.auto, self.cache = False, {}   # 10-22 (Neuro 73, Glenn): router = "auto" picks the brain per question

    @property
    def label(self):
        return "auto" if self.auto else self.on

    def pick(self, question):
        """10-22 router (multi-brain option 1): the brain this question needs, from the same guards that answer PY_OFF /
        HEALTH_OFF; anything else runs on the base (None). Brains are never merged together (two summed = 0% / 19%)."""
        import guards
        if "python" in self.brains and guards.needs_python(question):
            return "python"
        if "health" in self.brains and guards.needs_health(question):
            return "health"
        return None

    def _adapter(self, label):
        if label in self.cache:
            return self.cache[label]
        ad = torch.load(latest(self.brains[label], "adapter_step_*.pt"), map_location="cpu", weights_only=False)
        base = str(ad.get("base", ""))
        try:
            own = os.path.basename(latest(self.base_run))
        except FileNotFoundError:
            own = None
        # n76: a weight-only quantized copy (temp/n76_w8.py) names its source in QUANTIZED_FROM; brains of the source
        # run on it (brain quizzes re-checked on the copy, logs/n76_w8_brains.log)
        qf = os.path.join(CK, self.base_run, "QUANTIZED_FROM")
        src = open(qf).read().strip() if os.path.exists(qf) else None
        if base and os.path.basename(base.rstrip("/")) not in (self.base_run, own, src):
            raise ValueError(f"brain '{label}' was trained on '{base}', not on the loaded model '{self.base_run}' "
                             f"(brains do not transfer between checkpoints)")
        self.cache[label] = ad              # kept on the CPU: auto mode switches brains often
        return ad

    @torch.no_grad()
    def set(self, label):
        """label None / 'none' = no brain; 'auto' = the router (base until a question needs a brain). Returns the
        active label."""
        if label == "auto":
            self.set(None)
            self.auto = bool(self.brains)
            return self.label
        self.auto = False
        return self._set(label)

    @torch.no_grad()
    def _set(self, label):
        if label == self.on:
            return label
        if self.on is not None:                                            # restore first: never stack two brains
            for name, w in self.saved.items():
                self.model.get_submodule(name).weight.copy_(w.to(self.model.get_submodule(name).weight.device))
            self.saved, self.on = {}, None
        if label in (None, "", "none"):
            return None
        if label not in self.brains:
            raise KeyError(f"unknown brain '{label}' (have {sorted(self.brains)})")
        ad = self._adapter(label)
        scale = float(ad["lora_alpha"]) / float(ad["lora_r"])
        prefixes = sorted({k[: -len(".lora_A")] for k in ad["lora"] if k.endswith(".lora_A")})
        for p in prefixes:
            lin = self.model.get_submodule(p)
            A, B = ad["lora"][p + ".lora_A"], ad["lora"][p + ".lora_B"]
            self.saved[p] = lin.weight.detach().cpu().clone()
            from prune_ffn import lora_delta               # n78: a pruned FFN Linear takes the delta's kept rows / cols
            dev = lin.weight.device                         # n78: B @ A on the model's device (CPU: ~4 s per switch)
            lin.weight.add_(lora_delta(lin, scale * (B.to(dev).float() @ A.to(dev).float())).to(lin.weight.dtype))
        self.on = label
        return label


class Engine:
    def __init__(self, run, device="cuda", brains=(), allow_mismatch=False, prompt_budget=512, warm_grammar=True):
        if device == "cpu":
            patch_cpu_lif()
        from train_gpu import build_model
        import tool_data, grammar as G, tools_exec
        self.tool_data, self.G, self.device, self.run_name = tool_data, G, device, run
        blob = torch.load(latest(run), map_location="cpu", weights_only=False, mmap=True)
        self.c = c = dict(blob["cfg"])
        self.model = build_model(c).to(device)
        miss, unexp = self.model.load_state_dict(blob["model"], strict=False)
        if (miss or unexp) and not allow_mismatch:
            raise RuntimeError(f"{run}: checkpoint and its config disagree: {len(miss)} missing (e.g. {miss[:2]}), "
                               f"{len(unexp)} unexpected (e.g. {unexp[:2]}); refusing to serve a half-loaded model")
        self.model.eval()
        self.pruned = 0                     # n78 10-25: a pruned copy (prune_ffn.py) = the source + PRUNE_KEEP.pt
        if os.path.exists(os.path.join(CK, run, "PRUNE_KEEP.pt")):
            from prune_ffn import apply_prune
            self.pruned = apply_prune(self.model, torch.load(os.path.join(CK, run, "PRUNE_KEEP.pt")))
        if c.get("lut_subword_fallback", False):
            import lut_spell
            self.encode = lut_spell.make_encoder(c["vocab"], subword_fallback=True)
            self.decode = lut_spell.make_decoder(c["vocab"])
        else:
            from chat import load_tokenizer
            self.encode, self.decode = load_tokenizer(c["vocab"])
        self.tracer = Tracer(self.model, c)
        self.brain = BrainSlot(self.model, run, dict(brains))
        self.tools = tool_data.load_tools()
        self.offered = [t["name"] for t in self.tools]
        self.says = {t["name"]: t.get("say", "") for t in self.tools}
        self.proto = int(c.get("tool_proto", 1))
        self.grammar = G.ToolGrammar(self.offered, self.tools, proto=self.proto)
        self.table = G.TokenTable(c["vocab"])
        self.budget, self.lock = int(prompt_budget), threading.Lock()
        self.search_bias = float(os.environ.get("DEMO_SEARCH_BIAS", "1.0"))   # n66: 0 = off
        self.history = os.environ.get("DEMO_HISTORY", "0") == "1"             # n66: chat history in the prompt
        self.no_repeat = int(os.environ.get("DEMO_NO_REPEAT", "4"))            # n68: no-repeat n-gram size, 0 = off
        self.turns = []                     # [(question, final reply)] for the assistant history (D5)
        self.raw_history, self.last = "", None
        self.info = dict(run=run, device=device, tokens_trained=blob.get("tokens", 0), base_tokens=_base_tokens(run),
                         step=blob.get("step", 0), pruned_params=self.pruned,
                         params=sum(p.numel() for p in self.model.parameters()), n_layers=len(self.model.blocks),
                         attn_layers=len(self.tracer.attn), spiking_layers=len(self.tracer.spk_blocks),
                         attn_chunk=self.tracer.C, block_size=c["block_size"], spike_coded=bool(c.get("spike_code")),
                         tool_proto=self.proto, brains=sorted(self.brain.brains), h0_bins=H0_BINS,
                         prompt_budget=self.budget, search_bias=self.search_bias, history=self.history,
                         no_repeat=self.no_repeat, home=tools_exec.HOME)
        del blob
        # n64 10-15 (Glenn: choice B): untraced decode steps replay a CUDA graph of a right-padded forward (one graph
        # per (length bucket, ablate), captured on first use). Graph == eager on the same padded input (diff 0,
        # temp/n64_graphdiff.py); the meter stays on eager traced passes (prompt at step 0, replay after the answer).
        self.use_graph = device.startswith("cuda") and os.environ.get("DEMO_GRAPH", "1") != "0"
        self.graphs, self.graph_pool = {}, None
        # 10-21: graph length step. Answers are ~100-250 tokens, so with 512 each decode step of the answer pass computes
        # a 512-token forward. Measured (temp/n72_bucket.py, 5080, 8 questions): 256 = 23.5 s vs 512 = 35.1 s, 6/8
        # replies identical, 2 long ones differ late (padding length changes bf16 logits). Default 256; DEMO_BUCKET=512
        self.bucket = int(os.environ.get("DEMO_BUCKET", "256"))
        self._forced_cache = {}             # n68: grammar node -> its single allowed token (or None)
        self.trace_step0 = True             # n78: False = no traced pass at step 0 when it traces nothing
        self.defer_prompt_trace = False     # n78: True = prompt traces from the post-answer replay (StreamEngine)
        if warm_grammar:                    # the first visit of a DFA node walks the whole vocab: do it up front
            threading.Thread(target=self._warm, daemon=True).start()

    def warmup(self):
        """n68 10-18: pay the one-time costs at start-up instead of in the first answers: CUDA-graph capture of the
        512-token bucket (memory on / off), the WordNet dictionary (6 s cold) and the Spanish translator (16 s for
        the first translation). Takes the engine lock for the GPU part, so a request simply waits for it."""
        import tools_exec, guards
        if self.use_graph:
            with self.lock:
                for ab in (False, True):
                    for T in sorted({8, 2 * self.bucket - 8} if self.bucket < 512 else {8}):   # 10-21: + the 512 bucket
                        self._graph_logits([1] * T, ab)                                   # (prompts with history)
        tools_exec.define("test")
        guards.fix_spelling("warming", "warming")
        tools_exec.translate("hello", "Spanish")   # Spanish only: each translator holds ~0.3 GB of the RAM the trainer needs

    def _warm(self):
        for n in range(len(self.grammar.nodes)):
            if self.grammar.nodes[n]["kind"] != self.G.FREE:
                self.table.mask(self.grammar, n)

    # ---------------------------------------------------------------- decoding
    @torch.no_grad()
    def _graph_logits(self, ctx, ablate):
        """Last-position logits of `ctx` by CUDA-graph replay (input right-padded to a 512-multiple bucket; causal
        model, so the padding cannot reach position T-1). None if capture fails -> the caller runs eager."""
        T, bs = len(ctx), int(self.c["block_size"])
        B = self.bucket
        L = min(bs, -(-T // B) * B)
        key = (L, bool(ablate))
        if key not in self.graphs:
            static = torch.zeros(1, L, dtype=torch.long, device=self.device)
            fw = lambda: (lambda o: o[0] if isinstance(o, tuple) else o)(self.model(static, ablate_memory=ablate))
            try:
                st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st):
                    for _ in range(3):
                        fw()
                torch.cuda.current_stream().wait_stream(st)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.graph_pool):
                    out = fw()
                self.graph_pool = g.pool()
                self.graphs[key] = (g, static, out)
            except Exception as e:                      # e.g. a host sync inside the forward: stay eager for good
                print(f"[engine] CUDA graph capture failed at L={L}: {type(e).__name__}: {str(e)[:160]}; eager decode")
                self.use_graph = False
                return None
        g, static, out = self.graphs[key]
        static.zero_(); static[0, :T] = torch.tensor(ctx, device=self.device)
        g.replay()
        return out[0, T - 1].float().clone()

    @torch.no_grad()
    def _traced_logits(self, ctx, ablate):
        """Forward of `ctx` with the tracer on (the brain view / meter read it with tracer.per_position); returns the
        last position's logits. n78: one method, so engine_stream.py can replay it from a CUDA graph."""
        self.tracer.reset(len(ctx)); self.tracer.on = True
        try:
            lg = self.model(torch.tensor([ctx], device=self.device), ablate_memory=ablate)
        finally:
            self.tracer.on = False
        return (lg[0] if isinstance(lg, tuple) else lg)[0, -1].float()

    def _forced(self, grammar, node):
        """(token, next-node table) when grammar node `node` allows exactly one token, else None (cached per node)."""
        if grammar.nodes[node]["kind"] == self.G.FREE or grammar.accepting(node):
            return None
        key = (id(grammar), node)
        if key not in self._forced_cache:
            m, nx = self.table.mask(grammar, node)
            nz = m.nonzero().flatten()
            self._forced_cache[key] = (int(nz[0]), nx) if len(nz) == 1 else None
        return self._forced_cache[key]

    @torch.no_grad()
    def _decode(self, ids, n_trace, max_new, emit, ablate, temperature=0.0, seed=0, grammar=None, phase="answer",
                stop_eos=True, force_answer=False, stop_on_answer=False):
        """Generate up to max_new tokens after `ids` (grammar-constrained if given), tracing every forward.
        The prompt's last n_trace tokens are emitted with their traces first. Returns the generated text."""
        bs = int(self.c["block_size"]); gen = torch.Generator().manual_seed(seed)
        out, node, EOS = [], (grammar.root if grammar else None), self.tool_data.EOS_MARK
        live = not getattr(self, "trace_after", True)   # n63 10-15: trace once after the answer (same traces: causal
        tstr = lambda t: self.table.strs[t] if grammar else self.decode([t])   # model, full re-forward per token)
        n_txt = 0
        # n78 10-25 (Glenn: yes): defer_prompt_trace = the prompt's traces come from the post-answer replay pass (which
        # covers prompt + answer anyway) instead of a traced pass before decoding: one pass less per turn, the first word
        # sooner; the prompt's raster columns arrive after the decoded text (same order of columns)
        defer = self.defer_prompt_trace and not live
        for step in range(max_new + 1):
            ctx = (ids + out)[-bs:]
            T = len(ctx)
            # n78: a step 0 with nothing to trace (n_trace 0: the answer pass) may skip the traced pass (StreamEngine)
            traced = live or (step == 0 and ((n_trace > 0 and not defer) or self.trace_step0))
            # n68 10-18: a grammar node that allows exactly ONE token (JSON punctuation, key names, the rest of a tool
            # name) needs no forward pass: argmax over a single allowed token is that token (exact, also when sampling)
            forced = self._forced(grammar, node) if (grammar is not None and not traced and step < max_new) else None
            logits = None
            if forced is None:
                logits = self._graph_logits(ctx, ablate) if (self.use_graph and not traced) else None
                if logits is None and traced:
                    logits = self._traced_logits(ctx, ablate)
                elif logits is None:
                    logits = self.model(torch.tensor([ctx], device=self.device), ablate_memory=ablate)
                    logits = (logits[0] if isinstance(logits, tuple) else logits)[0, -1].float()
            if step == 0:
                k = min(n_trace, T)
                for tid, tr in (zip(ctx[T - k:], self.tracer.per_position(T - k)) if traced else ()):
                    emit(dict(type="token", kind="prompt", phase=phase, text=self.decode([tid]), **tr))
            elif live:
                emit(dict(type="token", kind="gen", phase=phase, text=tstr(out[-1]), **self.tracer.per_position(T - 1)[0]))
            else:
                emit(dict(type="text", kind="gen", phase=phase, text=tstr(out[-1]))); n_txt += 1
            if step == max_new:
                break
            if forced is not None:
                tok, nxt = forced
            else:
                if grammar is not None:
                    mask, nxt = self.table.mask(grammar, node, device=logits.device)
                    logits = logits.masked_fill(~mask, float("-inf"))
                    if step == 0 and (self.search_bias or force_answer) and node == grammar.root:
                        # n66 10-16: decode-time search bias on the TOOL-vs-ANSWER choice (temp/n66_bias.py; B4 +1:
                        # P(search|unknown) .693 -> .853, tool / refuse unchanged; +2 costs refusals)
                        tt = [t for t in mask.nonzero().flatten().tolist() if self.table.strs[t] and
                              (self.tool_data.MODE_TOOL.startswith(self.table.strs[t]) or self.table.strs[t].startswith(self.tool_data.MODE_TOOL))]
                        # n68 10-18: force_answer = the backend dropped an unrelated tool call -> ANSWER only
                        logits[tt] = float("-inf") if force_answer else logits[tt] + self.search_bias
                else:
                    logits[0] = float("-inf")
                # 2026-10-18 (Neuro 68): no-repeat n-gram on generated text (decide-phase free text and answers; the raw
                # memory showcase is untouched): the 1B model loops ("latest news, updates, and news ..."). DEMO_NO_REPEAT=0
                # off. Whole words only (tokens starting with a space): sub-word pieces of a spelled word are never banned.
                nr = self.no_repeat
                if (nr and phase != "raw" and len(out) >= nr and (grammar is None or grammar.nodes[node]["kind"] == self.G.FREE)
                        and all(tstr(t).startswith(" ") for t in out[-(nr - 1):])):
                    pre = tuple(out[-(nr - 1):])
                    ban = [out[i + nr - 1] for i in range(len(out) - nr + 1)
                           if tuple(out[i:i + nr - 1]) == pre and tstr(out[i + nr - 1]).startswith(" ")]
                    if ban:
                        logits[ban] = float("-inf")
                if temperature and temperature > 0:
                    tok = int(torch.multinomial(F.softmax(logits.cpu() / temperature, -1), 1, generator=gen))
                else:
                    tok = int(logits.argmax())
            out.append(tok)
            if grammar is not None:
                node = int(nxt[tok])
                if grammar.accepting(node):
                    break
                if stop_on_answer and "".join(self.table.strs[t] for t in out).lstrip().startswith(self.tool_data.MODE_ANSWER):
                    break
                if grammar.nodes[node]["kind"] == self.G.FREE and EOS in "".join(self.table.strs[t] for t in out[-4:]):
                    break
            elif stop_eos and (EOS in self.decode(out[-4:]) or self.decode(out[-2:]).endswith("###")):
                break
        kp = min(n_trace, len((ids + out[:n_txt])[-bs:]) - n_txt) if defer else 0   # deferred prompt traces
        if n_txt or kp:                                 # one traced pass over prompt + answer -> the brain-view replay
            full = (ids + out[:n_txt])[-bs:]; T = len(full)
            self._traced_logits(full, ablate)
            recs = self.tracer.per_position(T - n_txt - kp)
            for tid, tr in zip(full[T - n_txt - kp:T - n_txt], recs[:kp]):
                emit(dict(type="token", kind="prompt", phase=phase, text=self.decode([tid]), **tr))
            for tid, tr in zip(full[T - n_txt:], recs[kp:]):
                emit(dict(type="token", kind="gen", phase=phase, replay=True, text=tstr(tid), **tr))
        text = "".join(self.table.strs[t] for t in out) if grammar else self.decode(out)
        for cut in ((EOS, "###") if stop_eos else ()):
            if cut in text:
                text = text.split(cut)[0]
        self._last_out = list(out)          # n78: the generated token ids (engine_stream keeps raw text as tokens)
        return text.strip()

    # ---------------------------------------------------------------- assistant (protocol v2 tool loop)
    def _assistant_prompt(self, question):
        td = self.tool_data
        block = td.render_tools_block(self.offered, self.tools) + "\n\n"
        cur = f"### Instruction:\n{question.strip()}\n\n### Response:\n"
        # n66 10-16: history OFF by default -- SFT is single-turn, and past turns written as "ANSWER ..." make the model
        # copy ANSWER and stop calling tools after ~2 turns (temp/n66_hist.py). DEMO_HISTORY=1 brings it back.
        # 10-21 (chat brain step 1): earlier turns go in a "### Conversation:" block as User: / Assistant: lines (no
        # ANSWER / TOOL words to copy); the current turn stays in the trained single-turn format.
        # History keeps 128 tokens of the budget free, so a 512 budget leaves room for a short answer in the 512-token
        # graph bucket (each decode step re-runs the whole prompt).
        # The block is sft_data.render_conversation, the same function the chat brain's training data uses.
        from sft_data import render_conversation
        turns = list(self.turns) if (self.history or self.brain.on == "chat") else []   # the chat brain = history on
        while True:
            hist = render_conversation(turns)
            ids = self.encode(block + hist + cur)
            if len(ids) <= self.budget - (128 if turns else 0) or not turns:
                self._hist = render_conversation(turns[-1:])               # the tool-result pass: the last turn only
                self._ctx = " ".join(f"{q} {a}" for q, a in turns)          # what the guards may ground a follow-up on
                return ids, len(self.encode(cur)), len(turns)
            turns = turns[1:]                                              # oldest turn out first

    def assistant(self, question, emit, ablate=False, offline=False, temperature=0.0, seed=0, max_answer=120,
                  remember=True):
        import tools_exec, guards
        from sft_data import PROMPT_WITH_INPUT
        td = self.tool_data
        if self.brain.auto:                       # 10-22 router: switch before the prompt (the chat brain changes it)
            self.brain._set(self.brain.pick(question))
        ids, n_cur, in_view = self._assistant_prompt(question)
        emit(dict(type="turn", in_view=in_view, total_turns=len(self.turns), brain=self.brain.on, offline=offline,
                  prompt_tokens=len(ids), auto=self.brain.auto))
        canned = guards.social(question)          # n68: pure small talk -> fixed reply, labelled as the backend's
        if not canned and "python" in self.brain.brains and self.brain.on != "python" and guards.needs_python(question):
            canned = guards.PY_OFF                # 10-20 (Glenn): code requests with the Python brain off
        if not canned and "health" in self.brain.brains and self.brain.on != "health" and guards.needs_health(question):
            canned = guards.HEALTH_OFF            # 10-21 (Glenn): nutrient amounts with the health brain off
        if not canned and self.brain.on == "health" and guards.untaught_food(question):
            canned = guards.HEALTH_UNTAUGHT       # 10-20 (Glenn): a food the health brain was not taught
        if not canned and self.brain.on == "python" and guards.needs_python(question) and guards.python_untaught(question):
            canned = guards.PY_UNTAUGHT           # 10-21 (Glenn): a function the Python brain was not taught
        if not canned and guards.gibberish(question):
            canned = guards.NOT_UNDERSTOOD        # 10-21 (Glenn): mostly not English words
        if canned:
            if remember:
                self.turns.append((question, canned))
            out = dict(type="done", mode="BACKEND", reply=canned, tool=None, result=None)
            emit(out)
            return out
        # n68: when the backend would route this question anyway, a direct ANSWER is discarded -> stop decoding as soon
        # as the model picks ANSWER (same outcome, seconds faster: "Tell me about Marie Curie" decoded 120 tokens first)
        first = self._decode(ids, n_cur, 160, emit, ablate, temperature, seed, grammar=self.grammar, phase="decide",
                             stop_on_answer=guards.route(question) is not None)
        mode, call, body = td.parse_response(first)
        # 2026-10-18 demo guards (scripts/demo/guards.py), both shown on the page: an explicit request the model answered
        # from memory is routed to its tool; arguments the model mis-copied from the question are repaired.
        routed = False
        if mode == td.MODE_TOOL and call and guards.groundless(call, question, self._ctx):
            # a tool call with nothing to do with the message ("hi" -> translate "can you speak more slowly" to Arabic):
            # dropped, and the model answers instead (ANSWER forced at the mode choice)
            emit(dict(type="tool_dropped", name=call["name"], arguments=call.get("arguments", {})))
            first = self._decode(ids, 0, 160, emit, ablate, temperature, seed, grammar=self.grammar, phase="decide",
                                 force_answer=True, stop_on_answer=guards.route(question) is not None)
            mode, call, body = td.parse_response(first)
        if not (mode == td.MODE_TOOL and call):
            rc = guards.route(question)
            if rc:
                mode, call, routed = td.MODE_TOOL, rc, True
        result = None
        if mode == td.MODE_TOOL and call:
            say = "" if routed else first[len("TOOL"):].split("{")[0].strip()
            say = say or self.says.get(call["name"], "")
            fixed = guards.ground(call, question, offline, context=self._ctx)
            emit(dict(type="tool_call", say=say, name=call["name"], arguments=call.get("arguments", {}), routed=routed,
                      **({"corrected_from": fixed} if fixed else {})))
            result = tools_exec.run(call["name"], call.get("arguments", {}), offline=offline, question=question)
            emit(dict(type="tool_result", name=call["name"], **{k: v for k, v in result.items() if k != "card"},
                      card=result.get("card")))
            missing = guards.off_topic(question, result.get("text", "")) if (result.get("ok")
                                                                             and call["name"] == "search") else None
            if missing:                 # 10-21 (Glenn): the article found is about something else -> no made-up answer
                reply = (f"I didn't find that information. The closest article I found "
                         f"({str(result.get('text', '')).split(':')[0].strip()}) does not mention {missing}.")
            elif result.get("ok"):
                p2 = PROMPT_WITH_INPUT.format(instruction=question.strip(),
                                              input=f"Result from {call['name']}: {str(result.get('text', '')).strip()}")
                reply = self._decode(self.encode(self._hist + p2), 0, max_answer, emit, ablate, temperature, seed, phase="answer")
            elif call["name"] in ("search", "define", "get_weather"):   # 10-21 (Glenn): a lookup that found nothing
                reply = f"I didn't find that information ({str(result.get('text', 'no result')).strip()})."
            else:                       # a failed tool: say so plainly (the model would answer the question anyway)
                reply = f"Sorry, I could not get that: {str(result.get('text', 'the tool failed')).strip()}."
        elif mode == td.MODE_ANSWER:
            reply = body
        elif mode == td.MODE_TOOL:      # unparseable / cut-off call (grammar budget hit)
            reply = "Sorry, I could not complete that request."
        else:
            reply = first
        if not str(reply).strip():
            reply = "Sorry, I don't have an answer for that."
        reply, spelled = guards.fix_spelling(reply, question + " " + str((result or {}).get("text", "")),
                                             str((result or {}).get("text", "")) if (result or {}).get("ok") else "")
        if remember:
            self.turns.append((question, reply))
        out = dict(type="done", mode=mode, reply=reply, tool=call["name"] if (mode == td.MODE_TOOL and call) else None,
                   result=result, spelling=spelled)
        emit(out)
        return out

    # ---------------------------------------------------------------- raw text (memory showcase)
    def raw(self, text, emit, ablate=False, max_tokens=60, temperature=0.0, seed=0, remember=True):
        hist = self.raw_history
        if hist and text[:1].isalnum():         # 10-20: a typed next word continues the text: " brown", the token it
            text = " " + text                   # has inside it (history and new text are encoded separately)
        ids = (self.encode(hist) if hist else []) + self.encode(text)
        bs = int(self.c["block_size"])
        if len(ids) > bs:                       # 10-21: the oldest text is cut silently otherwise (pasted word pairs too)
            emit(dict(type="note", msg=f"the text so far is {len(ids):,} tokens; the model reads the last {bs:,}, so the "
                                       f"first {len(ids) - bs:,} have dropped out. Reset to start over."))
        reply = self._decode(ids, len(self.encode(text)), max_tokens, emit, ablate, temperature, seed, phase="raw",
                             stop_eos=False)
        if remember:
            self.raw_history = hist + text + reply
        emit(dict(type="done", mode="RAW", reply=reply))
        return reply

    # -------------------------------------------------- memory test (one click; 10-20, Glenn)
    @torch.no_grad()
    def memtest(self, emit, N=8, trials=20, seed=0, offset=64):
        """The trainer's cross-chunk recall test (temp/xchunk_haystack_eval.py: final model 99-100% memory on, 1.6-12.5%
        off) as a demo: N word pairs [k1 v1 .. kN vN] at the start of attention chunk 1, ordinary text after them, one
        key at `offset` into chunk 2 -> which of the N values follows? Attention cannot see across the chunk boundary,
        so only the memory layer can carry the pairs. Each try runs memory on and off; one try is shown as words.
        Keys / values are the training's ranges (train_gpu.MQAR_*), text tokens equal to a key are replaced as there."""
        import random
        from train_gpu import MQAR_KEY_LO, MQAR_KEY_HI, MQAR_VAL_LO, MQAR_VAL_HI
        C = int(self.tracer.C)
        if C and self.tracer.W:                 # n80 10-26 (review fix c): a sliding window W sees keys q-W+1..q, so the
            C = max(C, int(self.tracer.W))      # query must sit past W + the pairs: chunk length >= W, offset >= the pairs
        offset = max(offset, 2 * N)
        if not C or C + offset + 1 > int(self.c["block_size"]):
            emit(dict(type="error", msg="this model has no attention chunks: the memory test needs them")); return
        rng, fill = random.Random(seed), self.encode(_FILLER)
        gap_n = C - 2 * N + offset
        hits, example = {False: 0, True: 0}, None
        for t in range(trials):
            keys = rng.sample(range(MQAR_KEY_LO, MQAR_KEY_HI), N)
            vals = rng.sample(range(MQAR_VAL_LO, MQAR_VAL_HI), N)
            s0, ks = rng.randrange(len(fill)), set(keys)
            gap = [fill[(s0 + i) % len(fill)] for i in range(gap_n)]
            gap = [g if g not in ks else rng.randrange(6000, 8000) for g in gap]
            qi = rng.randrange(N)
            x = torch.tensor([[v for kv in zip(keys, vals) for v in kv] + gap + [keys[qi]]], device=self.device)
            pick = {}
            for ab in (False, True):
                lg = self.model(x, ablate_memory=ab)
                lg = (lg[0] if isinstance(lg, tuple) else lg)[0, -1].float()
                pick[ab] = int(torch.argmax(lg[torch.tensor(vals, device=lg.device)]))
                hits[ab] += pick[ab] == qi
            if example is None:
                example = (keys, vals, qi, pick)
            emit(dict(type="memtest_progress", done=t + 1, trials=trials, on=hits[False], off=hits[True]))
        keys, vals, qi, pick = example
        w = lambda i: self.decode([i]).strip() or f"#{i}"
        emit(dict(type="memtest", N=N, trials=trials, chunk=C, gap=gap_n, pos=C + offset,
                  pairs=[[w(k), w(v)] for k, v in zip(keys, vals)], query=w(keys[qi]), answer=w(vals[qi]),
                  on_pick=w(vals[pick[False]]), off_pick=w(vals[pick[True]]),
                  on_ok=pick[False] == qi, off_ok=pick[True] == qi, on=hits[False], off=hits[True],
                  chance=round(100 / N, 1)))
        emit(dict(type="done", mode="MEMTEST", reply=""))

    def reset(self):
        self.turns, self.raw_history, self.last = [], "", None
