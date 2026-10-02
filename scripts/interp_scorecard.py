"""interp_scorecard.py -- 2026-10-02 (M1 in docs/interpretable_spiking_plan_2026_10_01.md; Glenn OK 10-02).
Interpretability scorecard: one comparable row per checkpoint, so "interpretable" is a number we track like val_ppl.

Metrics (higher = more interpretable unless noted):
  selectivity   per unit (split into +/- half-units): share of its active tokens taken by its single most common
                token (top1, median and p90) and 1 - H(token | active)/H(token) (ent). Spiking units are active
                where they fire; DSN units where they win the k-WTA / fire; dense MLP units on their top-p tokens
                (p = DENSE_P per sign, a fixed matched density).
  dead          share of units active on < MIN_ACTIVE tokens (lower = better)
  attn_breadth  keys needed for 90% of a query's attention mass, mean over heads/queries at positions >= 64
                (lower = easier to audit); attn_sink = mean attention mass on position 0
  circuit       (position, unit) entries needed for 90% of |activation x gradient| attribution of the target logit,
                median over examples (lower = more compact circuit): MQAR recall (N=8) and next-token LM
  mem_trace     memory layers only: share of MQAR queries whose read is dominated by the ONE write that stored the
                queried pair (exact decomposition of r_t = sum_s ((prod alpha) * kn_s . q_t) u_s, with u_s ~ w_s*v_s,
                i.e. ignoring the erase correction); at gap 0 and gap 192 (real text in between); + top-1 write share
  events        op model per token (seq 512 / 4096): share of ops triggered by spikes / k-WTA winners (event_share)
                and share that are pure accumulates (ac_share; ternary-spike inputs)
  autointerp    optional (--autointerp K, GPU): Llama-3.2-3B-Instruct explains K sampled units per layer from their top
                contexts, then judges 5 held-out activating + 5 random contexts; score = balanced accuracy
Usage:  python scripts/interp_scorecard.py [--device cpu|cuda] [--autointerp 0] name=ckpt.pt [name=ckpt.pt ...]
Appends one JSON line per model to logs/scorecard.jsonl. CPU mode swaps the LIF Triton kernels for a torch
reference (same forward, tanh surrogate gradient)."""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import lif_exact as LE

MIN_ACTIVE, DENSE_P = 20, 0.05
OUT = "/home/glenn/projects/neuro/logs/scorecard.jsonl"


def lif_ml_ref(cur, beta, vth=1.0, K=None, alpha=2.0, gamma=1.0, levels=1, signed=False):
    B, T, N = cur.shape
    beta = beta if torch.is_tensor(beta) else torch.full((N,), float(beta), device=cur.device)
    mem = torch.zeros(B, N, device=cur.device); outs = []
    for t in range(T):
        pre = beta * mem + cur[:, t].float()
        hard = torch.clamp(torch.floor(pre / vth), 0, levels)
        if signed:
            hard = hard - torch.clamp(torch.floor(-pre / vth), 0, levels)
        outs.append(hard + (torch.tanh(pre) - torch.tanh(pre).detach()))
        mem = pre - gamma * hard * vth
    return torch.stack(outs, 1).to(cur.dtype), mem.to(cur.dtype)


def load(path, device):
    from train_gpu import build_model
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c); m.load_state_dict(s["model"], strict=False); m.to(device).eval()
    return m, c


class Capture:
    """Wraps every block: records the block input, H-block spikes, and the hidden units of DSN / MLP FFNs."""
    def __init__(self, m):
        self.m, self.acts, self.inputs, self.grad = m, {}, {}, False
        for i, b in enumerate(m.blocks):
            orig = b.forward_sequence
            def fs(x_seq, *a, _o=orig, _i=i, _b=b, **k):
                self.inputs[_i] = x_seq.detach()
                out = _o(x_seq, *a, **k)
                if not hasattr(_b, "qkv"):
                    spk = out[3]
                    if self.grad and spk.requires_grad:
                        spk.retain_grad()
                    self.acts[_i] = ("spk", spk)
                return out
            b.forward_sequence = fs
            if hasattr(b, "qkv") and b.ffn is not None:
                last = b.ffn.w2 if hasattr(b.ffn, "w2") else b.ffn[-1]
                kind = "dsn" if hasattr(b.ffn, "k_cur") else "mlp"
                def pre(mod, inp, _i=i, _k=kind):
                    h = inp[0]
                    if self.grad and h.requires_grad:
                        h.retain_grad()
                    self.acts[_i] = (_k, h)
                last.register_forward_pre_hook(pre)


def val_batch(c, B, T, seed=0):
    from train_gpu import StreamText, BIN
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                      val_tokens=c.get("data_val_tokens"))
    np.random.seed(seed)
    x, y = data.batch(B, T, "val", "cpu")
    return data, x, y


def mqar(data, N, g, B, gen):
    from train_gpu import MQAR_KEY_LO, MQAR_KEY_HI, MQAR_VAL_LO, MQAR_VAL_HI
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


def selectivity(acts, toks, H_all, firing):
    rows = {}
    for i, (kind, a) in sorted(acts.items()):
        a = a.detach().float().reshape(-1, a.shape[-1]).cpu()
        if kind in ("spk", "dsn"):
            masks = [a > 0, a < 0]
            firing[i] = (kind, (a != 0).float().mean().item())
        else:
            masks = []
            for sgn in (1, -1):
                k = max(1, int(round(DENSE_P * a.shape[0])))
                thr = torch.topk(sgn * a, k, dim=0).values[-1]
                masks.append((sgn * a) >= thr)
            firing[i] = (kind, 1.0)
        sc, n_units = [], a.shape[1]
        dead = ((masks[0].sum(0) < MIN_ACTIVE) & (masks[1].sum(0) < MIN_ACTIVE)).float().mean().item()
        for mk in masks:
            for u in range(n_units):
                idx = mk[:, u].nonzero().squeeze(1)
                if idx.numel() < MIN_ACTIVE:
                    continue
                _, cnt = torch.unique(toks[idx], return_counts=True)
                p = cnt.float() / idx.numel()
                sc.append((p.max().item(), 1 + (p * p.log()).sum().item() / H_all))
        sc = np.array(sc) if sc else np.zeros((0, 2))
        rows[i] = dict(kind=kind, units=n_units, scored=len(sc), dead=round(dead, 4),
                       top1_med=round(float(np.median(sc[:, 0])), 4) if len(sc) else None,
                       top1_p90=round(float(np.quantile(sc[:, 0], 0.9)), 4) if len(sc) else None,
                       ent_med=round(float(np.median(sc[:, 1])), 4) if len(sc) else None)
    return rows


def attention_breadth(m, cap):
    out = {}
    for i, b in enumerate(m.blocks):
        if not hasattr(b, "qkv") or i not in cap.inputs:
            continue
        x = cap.inputs[i]
        if getattr(b, "attn_canon", 0):
            k = b.attn_canon
            x = x + F.conv1d(F.pad(x.transpose(1, 2), (k - 1, 0)), b.canon_w.to(x.dtype), groups=x.size(-1)).transpose(1, 2)
        with torch.no_grad():
            W = b._attn_weights(b.norm_attn(x)).float()          # (B, H, T, T)
        W = W[:, :, 64:, :]
        srt = W.sort(-1, descending=True).values.cumsum(-1)
        n90 = (srt < 0.9).sum(-1).float() + 1
        out[i] = dict(breadth90=round(n90.mean().item(), 2), sink=round(W[..., 0].mean().item(), 4),
                      topk=getattr(b, "attn_topk", 0))
    return out


def circuit(m, cap, x, target_ids, device):
    """Per-example count of (pos, unit) attribution entries covering 90% of |act * grad| of the target logit."""
    cap.grad = True; cap.acts.clear()
    with torch.enable_grad():
        lg = m(x.to(device))
        lg = lg[0] if isinstance(lg, tuple) else lg
        tgt = lg[:, -1, :].float().gather(1, target_ids.to(device).unsqueeze(1)).sum()
        tgt.backward()
    per_ex = []
    attrs = [(a.detach() * a.grad).abs().float() for _, a in cap.acts.values() if a.grad is not None]
    cap.grad = False
    m.zero_grad(set_to_none=True)
    if not attrs:
        return None
    for b in range(x.shape[0]):
        v = torch.cat([a[b].reshape(-1) for a in attrs])
        v = v[v > 0]
        if v.numel() == 0:
            continue
        s = v.sort(descending=True).values.cumsum(0)
        per_ex.append(((s < 0.9 * s[-1]).sum().item() + 1, v.numel()))
    n = np.array(per_ex)
    return dict(entries90_med=int(np.median(n[:, 0])), active_entries_med=int(np.median(n[:, 1])),
                frac_of_active=round(float(np.median(n[:, 0] / n[:, 1])), 4))


def mem_trace(m, data, device, N=8, gaps=(0, 192), B=64):
    hblocks = [b for b in m.blocks if getattr(b, "gdn2_enabled", False) and not hasattr(b, "qkv")]
    if not hblocks:
        return None
    for b in hblocks:
        b.audit = True
    res = {}
    for g in gaps:
        gen = torch.Generator().manual_seed(1000 * N + g); np.random.seed(1000 * N + g)
        x, vals, qi = mqar(data, N, g, B, gen)
        with torch.no_grad():
            lg = m(x.to(device)); lg = lg[0] if isinstance(lg, tuple) else lg
        correct = (lg[:, -1, :].float().cpu().gather(1, vals).argmax(1) == qi)
        a = hblocks[0]._audit
        if a["k"].shape[0] != x.shape[0]:
            res[g] = None; continue                               # multi-head memory: not supported yet
        k = a["k"].float(); kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        q, alpha = a["q"].float(), a["a"].float().clamp_min(1e-12)
        u = (a["w"].float() * a["v"].float()).norm(dim=-1)      # (B, T) size of each write (erase term ignored)
        t = x.shape[1] - 1
        la = torch.log(alpha)
        # retention of write s at read t: prod_{r=s+1..t} alpha_r  (per channel)
        tail = la[:, :t + 1].flip(1).cumsum(1).flip(1)            # sum_{r=s..t} log alpha_r
        ret = torch.exp(tail - la[:, :t + 1])                     # drop r = s
        coef = (kn[:, :t + 1] * q[:, t:t + 1] * ret).sum(-1)      # (B, T') key.query per write, decayed
        score = coef.abs() * u[:, :t + 1]
        top = score.argmax(1).cpu()
        hit = (top == 2 * qi + 1)
        share = (score.max(1).values / score.sum(1).clamp_min(1e-12)).cpu()
        res[g] = dict(trace=round(hit.float().mean().item(), 4),
                      trace_given_correct=round(hit[correct].float().mean().item(), 4) if correct.any() else None,
                      top1_share=round(share.mean().item(), 4), recall_acc=round(correct.float().mean().item(), 4))
    for b in hblocks:
        b.audit = False; b.__dict__.pop("_audit", None)
    return res


def op_model(m, c, firing, seq_len):
    """Per-token ops. mac = dense multiply-accumulates; ev = ops executed only for active units / spikes;
    ac = the subset of ev that are pure accumulates (inputs are ternary spikes)."""
    d, dm, V = m.d, m.d_mem, m.vocab
    mac = ev = ac = 0.0
    for i, b in enumerate(m.blocks):
        if hasattr(b, "qkv"):
            mac += 4 * dm * dm + (dm * dm if getattr(b, "attn_gate", None) is not None else 0)
            k = getattr(b, "attn_topk", 0)
            mac += dm * seq_len + dm * (min(k, seq_len) if k else seq_len)   # scores + weighted values
            f = b.ffn
            if f is None:
                continue
            if hasattr(f, "k_cur"):                                           # DSN
                H = f.w1.out_features; r = firing.get(i, ("dsn", f.k / H))[1]
                mac += dm * H                                                 # apical gate (dense)
                ev += 2 * r * H * dm                                          # basal payload + down-proj, winners only
            else:
                H = f.w2.in_features if hasattr(f, "w2") else f[-1].in_features
                mac += (3 if hasattr(f, "w2") else 2) * dm * H
            continue
        N = b.n_neurons; r = firing.get(i, ("spk", 0.1))[1]
        mac += (d if i == 0 else dm) * N + 3 * dm * dm                        # dense input + memory state ops
        gr = c.get("gate_rank", 0)
        n_full = 4 + (0 if gr else 3)                                         # k, v, q, ff (+ full-rank gates)
        ev += r * n_full * N * dm; ac += r * n_full * N * dm
        if gr:
            ev += r * 3 * N * gr; ac += r * 3 * N * gr; mac += 3 * gr * dm   # low-rank gates
    mac += dm * V
    if getattr(m, "engram_v2_enabled", False):
        mac += m.engram_v2.proj.in_features * d                              # Engram: lookups + one projection
    tot = mac + ev
    return dict(event_share=round(ev / tot, 4), ac_share=round(ac / tot, 4), mops=round(tot / 1e6, 2))


def autointerp(m, c, cap, device, k_units, seed=0):
    """Explain-then-detect autointerp with Llama-3.2-3B-Instruct (GPU). Returns mean balanced accuracy per layer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sys as _s
    _s.path.insert(0, "/home/glenn/projects/neuro/scripts")
    from build_corpus_v4 import load_tok
    lut = np.load(f"/home/glenn/projects/neuro/data/vocab_map_{c['vocab']}.npy")
    inv = np.zeros(c["vocab"], dtype=np.int64); nz = np.nonzero(lut)[0]; inv[lut[nz]] = nz
    btok = load_tok()
    name = "meta-llama/Llama-3.2-3B-Instruct"
    tk = AutoTokenizer.from_pretrained(name, local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval()
    data, x, _ = val_batch(c, 32, 512, seed=seed + 1)
    cap.acts.clear()
    with torch.no_grad():
        m(x.to(device))
    rng = np.random.default_rng(seed)
    yes, no = tk.encode(" Yes", add_special_tokens=False)[-1], tk.encode(" No", add_special_tokens=False)[-1]
    def ctx(b, t):
        ids = x[b, max(0, t - 12):t + 1].tolist()
        pre = btok.decode([int(inv[i]) for i in ids[:-1]]); cur = btok.decode([int(inv[ids[-1]])])
        return f"{pre}<<{cur}>>"
    def chat_ids(prompt):
        enc = tk.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True)
        return enc["input_ids"].to(device)
    def ask(prompt, max_new=40):
        ids = chat_ids(prompt)
        with torch.no_grad():
            return lm.generate(ids, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tk.eos_token_id)[0, ids.shape[1]:], ids
    out = {}
    for i, (kind, a) in sorted(cap.acts.items()):
        a = a.detach().float()
        act = a.abs() if kind in ("spk", "dsn") else a.clamp_min(0)
        flat = act.reshape(-1, act.shape[-1])
        alive = (flat > 0).sum(0) >= 15 if kind != "mlp" else torch.ones(flat.shape[1], dtype=torch.bool, device=flat.device)
        cand = alive.nonzero().squeeze(1).cpu().numpy()
        if len(cand) == 0:
            continue
        scores = []
        for u in rng.choice(cand, size=min(k_units, len(cand)), replace=False):
            v = flat[:, u].cpu(); order = torch.argsort(v, descending=True)
            pos = [int(p) for p in order[:15] if v[p] > 0]
            if len(pos) < 10:
                continue
            ex = [ctx(p // 512, p % 512) for p in pos[:10]]
            held = [ctx(p // 512, p % 512) for p in pos[10:15]]
            neg_pool = (v == 0).nonzero().squeeze(1).numpy() if kind != "mlp" else order[len(order) // 2:].numpy()
            neg = [ctx(int(p) // 512, int(p) % 512) for p in rng.choice(neg_pool, size=len(held), replace=False)]
            gen, _ = ask("These text snippets all activate one neuron at the token marked <<like this>>:\n" +
                         "\n".join(f"- {e}" for e in ex) +
                         "\nIn at most 12 words, what does this neuron detect? Answer with the description only.")
            expl = tk.decode(gen, skip_special_tokens=True).strip()
            hits = []
            for txt, lab in [(h, 1) for h in held] + [(n_, 0) for n_ in neg]:
                ids = chat_ids(f"A neuron detects: {expl}\nText: {txt}\nDoes the token marked <<like this>> match? "
                               f"Answer Yes or No.")
                with torch.no_grad():
                    lgt = lm(ids).logits[0, -1].float()
                hits.append(int((lgt[yes] > lgt[no]).item() == bool(lab)))
            scores.append(np.mean(hits))
        out[i] = dict(kind=kind, units=len(scores), balanced_acc=round(float(np.mean(scores)), 4) if scores else None)
    del lm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--autointerp", type=int, default=0)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    if a.device == "cpu":
        LE.lif_exact_ml = lif_ml_ref
        LE.lif_exact = lambda cur, beta, vth=1.0, K=None, alpha=2.0, gamma=1.0: lif_ml_ref(cur, beta, vth, gamma=gamma)
        torch.set_num_threads(a.threads)
    for spec in a.models:
        name, path = spec.split("=", 1)
        t0 = time.time()
        m, c = load(path, a.device)
        cap = Capture(m)
        data, x, y = val_batch(c, a.B, a.T)
        with torch.no_grad():
            m(x.to(a.device))
        toks = x.reshape(-1)
        _, cnt = torch.unique(toks, return_counts=True); p = cnt.float() / toks.numel()
        H_all = -(p * p.log()).sum().item()
        firing = {}
        row = dict(name=name, ckpt=path, time=time.strftime("%Y-%m-%d %H:%M"), params_M=round(sum(
            q.numel() for q in m.parameters()) / 1e6, 2))
        row["selectivity"] = selectivity(cap.acts, toks, H_all, firing)
        row["attention"] = attention_breadth(m, cap)
        gen = torch.Generator().manual_seed(7)
        xq, vals, qi = mqar(data, 8, 0, 32, gen)
        row["circuit_mqar"] = circuit(m, cap, xq, vals[torch.arange(32), qi], a.device)
        _, xl, yl = val_batch(c, 8, 256, seed=3)
        row["circuit_lm"] = circuit(m, cap, xl, yl[:, -1], a.device)
        row["mem_trace"] = mem_trace(m, data, a.device)
        row["events"] = {L: op_model(m, c, firing, L) for L in (512, 4096)}
        row["firing"] = {i: round(v[1], 4) for i, v in firing.items()}
        if a.autointerp:
            cap.acts.clear()
            row["autointerp"] = autointerp(m, c, cap, a.device, a.autointerp)
        row["seconds"] = round(time.time() - t0, 1)
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")
        sel = row["selectivity"]
        print(f"== {name} ({row['params_M']}M, {row['seconds']}s)")
        for i, s in sel.items():
            print(f"  L{i} {s['kind']:3s} units {s['units']:5d} dead {s['dead']:.3f} | top1 med {s['top1_med']} "
                  f"p90 {s['top1_p90']} | ent med {s['ent_med']} | firing {row['firing'].get(i)}")
        for i, s in row["attention"].items():
            print(f"  L{i} attn breadth90 {s['breadth90']} sink {s['sink']} (topk {s['topk']})")
        print(f"  circuit mqar {row['circuit_mqar']} | lm {row['circuit_lm']}")
        print(f"  mem_trace {row['mem_trace']}")
        print(f"  events {row['events']}")
        if a.autointerp:
            print(f"  autointerp {row['autointerp']}")
        del m
        if a.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
