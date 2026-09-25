"""DPO (Direct Preference Optimization) trainer for PsycorNeuro.

Rafailov et al. 2023: skip the reward-model + RL of RLHF; directly optimize the
policy against a reference using a contrastive loss on chosen vs rejected pairs.

Loss (per sample):
    log_ratio_w  = log pi(y_w | x) - log pi_ref(y_w | x)
    log_ratio_l  = log pi(y_l | x) - log pi_ref(y_l | x)
    L = -log sigmoid( beta * (log_ratio_w - log_ratio_l) )

The reference policy pi_ref is a FROZEN copy of the SFT model. We keep two copies
of the model in memory (doubles VRAM); for our 30M screening this is fine.

Dataset: HuggingFaceH4/ultrafeedback_binarized by default — (prompt, chosen, rejected).

  python src/train_dpo.py --config configs/dpo_30m.yaml
"""
import os, sys, time, math, argparse, glob, copy
import numpy as np
import torch
import torch.nn.functional as F
import yaml
sys.path.insert(0, os.path.dirname(__file__))
from train_sft import (build_model, build_optimizers, load_lut_encode,
                        load_lut_decode, gen_check, save_ckpt,
                        make_encoder_pair)


def log1mexp(x):
    """Numerically stable log(1 - exp(x)) for x <= 0 (Mächler 2012). x is a
    length-normalized mean log-prob, so exp(x) in (0,1). Used by the ORPO
    log-odds term. Clamp just below 0 to avoid log(0) when the response is
    (near-)deterministic."""
    x = x.clamp(max=-1e-7)
    return torch.where(x > -0.6931471805599453,     # -ln 2
                       torch.log(-torch.expm1(x)),
                       torch.log1p(-torch.exp(x)))


def load_preferences_jsonl(paths, pair_types=None):
    """Local preference JSONL from scripts/gen_teacher_data.py: flat records of
    {prompt, chosen, rejected, pair_type}. Used by the why-narrated stage
    (idea #5) and the calibrated-abstention stage (idea #2).

    pair_types: optional allow-list of pair_type values, so a single generated
    file can be ablated (e.g. why_vs_rejected only, or the answerable half of
    the abstention set) without regenerating anything.
    """
    import json as _json
    if isinstance(paths, str):
        paths = [paths]
    out, counts, bad = [], {}, 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = _json.loads(line)
                except _json.JSONDecodeError:
                    bad += 1         # partial trailing line from a killed writer
                    continue
                pt = r.get("pair_type", "?")
                if pair_types and pt not in pair_types:
                    continue
                if not (r.get("prompt") and r.get("chosen") and r.get("rejected")):
                    bad += 1
                    continue
                out.append({"prompt": r["prompt"], "chosen": r["chosen"],
                            "rejected": r["rejected"]})
                counts[pt] = counts.get(pt, 0) + 1
    print(f"[prefs] local jsonl {paths}: {len(out)} pairs {counts} (bad {bad})")
    return out


def load_preferences(dataset="HuggingFaceH4/ultrafeedback_binarized",
                     split="train_prefs"):
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split)
    out = []
    for ex in ds:
        # UltraFeedback format: chosen/rejected are lists of {role, content} dicts
        try:
            prompt = ex.get("prompt") or ex["chosen"][0]["content"]
            chosen = ex["chosen"][-1]["content"]
            rejected = ex["rejected"][-1]["content"]
            out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        except (KeyError, IndexError, TypeError):
            continue
    return out


PROMPT_FMT = "### Instruction:\n{message}\n\n### Response:\n"
EOS_MARK = "\n###"


def encode_pair(prompt, response, encode_fn, max_len):
    """Returns (input_ids, response_mask). response_mask[t]=True for response tokens."""
    p_text = PROMPT_FMT.format(message=prompt.strip())
    r_text = response.strip() + EOS_MARK
    p_ids = encode_fn(p_text); r_ids = encode_fn(r_text)
    if not r_ids or len(p_ids) >= max_len - 1:
        return None
    full = (p_ids + r_ids)[:max_len]
    mask = [False] * len(p_ids) + [True] * len(r_ids)
    mask = mask[:max_len]
    return full, mask


def build_dpo_examples(raw, encode_fn, max_len):
    out = []; skipped = 0
    for ex in raw:
        try:
            w = encode_pair(ex["prompt"], ex["chosen"], encode_fn, max_len)
            l = encode_pair(ex["prompt"], ex["rejected"], encode_fn, max_len)
            if w is None or l is None:
                skipped += 1; continue
            out.append({"chosen": w, "rejected": l})
        except Exception:
            skipped += 1
    return out, skipped


def collate(batch, pad_id=0):
    """Pad chosen and rejected sequences. Returns dict of tensors."""
    seqs = [b["chosen"] for b in batch] + [b["rejected"] for b in batch]
    max_len = max(len(s[0]) for s in seqs)
    B2 = len(seqs)
    ids = torch.zeros(B2, max_len, dtype=torch.long)
    msk = torch.zeros(B2, max_len, dtype=torch.bool)
    nontrivial = torch.zeros(B2, dtype=torch.bool)
    for i, (x, m) in enumerate(seqs):
        n = len(x)
        ids[i, :n] = torch.tensor(x, dtype=torch.long)
        msk[i, :n] = torch.tensor(m, dtype=torch.bool)
        nontrivial[i] = True
    # Split back: chosen = first B rows, rejected = last B rows
    return ids, msk


def response_logp(model, ids, mask, vocab, average=False, return_stats=False):
    """Response-position log-prob per example, sum (DPO) or average (SimPO).

    Args:
      model:        SpikingHebbianLM (or LoRA-wrapped)
      ids, mask:    (B2, T) tokens and response-position mask
      vocab:        vocab size (unused; kept for backwards signature)
      average:      True -> return mean per-token logp per example (SimPO reward);
                    False -> return sum (DPO log pi(y|x)).
      return_stats: also return the model's spike rate (from return_stats=True)

    Returns: scalar-per-example (B2,) tensor of logp, and if return_stats,
    a scalar spike rate.
    """
    if return_stats:
        out = model(ids, return_stats=True)
        # Model may return (logits, sr) or (logits, sr, ...); take first two.
        logits, sr = out[0], out[1]
    else:
        logits = model(ids)
        sr = None
    logp = F.log_softmax(logits, dim=-1)
    # Shift: predict ids[t] from logits at [t-1]
    target = ids[:, 1:]                          # (B2, T-1)
    pred = logp[:, :-1, :]                       # (B2, T-1, V)
    m = mask[:, 1:]                              # (B2, T-1) — mark response positions
    chosen_logp = pred.gather(2, target.unsqueeze(-1)).squeeze(-1)   # (B2, T-1)
    chosen_logp = chosen_logp * m.float()
    total = chosen_logp.sum(dim=1)               # (B2,)
    if average:
        length = m.float().sum(dim=1).clamp_min(1.0)
        total = total / length
    if return_stats:
        return total, sr
    return total


def sum_response_logp(model, ids, mask, vocab):
    """Backwards-compat shim: sum-only, no stats. Kept so any external caller
    still resolves; new code should use response_logp() directly."""
    return response_logp(model, ids, mask, vocab, average=False, return_stats=False)


class DPOLoader:
    def __init__(self, examples, batch_size, seed=0):
        self.examples = examples
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(examples))
        self.cursor = 0
    def __next__(self):
        if self.cursor + self.batch_size > len(self.examples):
            self.order = self.rng.permutation(len(self.examples))
            self.cursor = 0
        batch_idx = self.order[self.cursor:self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return collate([self.examples[i] for i in batch_idx])
    def __iter__(self): return self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    with open(args.config) as f: c = yaml.safe_load(f)
    device = args.device
    torch.manual_seed(c.get("seed", 0))

    # Build policy model + load SFT init
    policy = build_model(c).to(device)
    sft_init = c["dpo_init"]
    if "/" not in sft_init:
        cks = sorted(glob.glob(
            f"/home/glenn/projects/neuro/models/checkpoints/{sft_init}/step_*.pt"))
        if not cks: raise FileNotFoundError(f"no SFT ckpt for {sft_init}")
        sft_init = cks[-1]
    blob = torch.load(sft_init, map_location=device, weights_only=False)
    policy.load_state_dict(blob["model"], strict=False)
    print(f"loaded SFT init: {sft_init} ({blob.get('tokens',0)/1e6:.0f}M tok)")

    # Reference model: frozen copy of policy at SFT init (no adapter — it IS the
    # DPO reference distribution pi_ref). Under LoRA, only the policy is adapted,
    # so the reference stays the plain frozen base.
    #
    # Objective selection. Back-compat: default is "simpo" if simpo_enabled was
    # set (old configs), else "dpo".
    #   dpo   (Rafailov 2023): reference-based; L = -log sig(beta*(dw - dl)).
    #   simpo (Meng 2024, 2405.14734): reference-free; length-normalized avg-logp
    #          reward + target margin gamma. L = -log sig(beta*(pi_w - pi_l) - gamma).
    #   orpo  (Hong 2024, 2403.07691): reference-free; SFT-NLL on chosen PLUS a
    #          weak odds-ratio penalty. Keeping the explicit maximize-chosen NLL
    #          term is exactly what defends against the likelihood-displacement
    #          that word-souped standard DPO/SimPO on this memory-heavy pipeline
    #          (neuro_dpo_retest_2026_08_26_null / neuro_simpo_collapse_2026_07_26).
    #          L = -mean_logp(y_w) + lambda * -log sig(log_odds(y_w) - log_odds(y_l)).
    objective = c.get("objective", "simpo" if c.get("simpo_enabled", False) else "dpo")
    assert objective in ("dpo", "simpo", "orpo"), f"unknown objective {objective!r}"
    simpo = (objective == "simpo")            # kept for downstream back-compat
    orpo = (objective == "orpo")
    reference_free = objective in ("simpo", "orpo")
    simpo_gamma = float(c.get("simpo_gamma", 0.5))   # target margin (simpo only)
    orpo_lambda = float(c.get("orpo_lambda", 0.1))   # odds-ratio weight (orpo only)
    if reference_free:
        reference = None
        extra = (f"gamma={simpo_gamma}" if simpo else f"lambda={orpo_lambda}")
        print(f"{objective.upper()}: reference-free (no reference model) | {extra}")
    else:
        reference = build_model(c).to(device)
        reference.load_state_dict(blob["model"], strict=False)
        for p in reference.parameters(): p.requires_grad_(False)
        reference.eval()

    # LoRA (parameter-efficient DPO): adapt only the policy. dpo_init must be a
    # PLAIN (merged) checkpoint — if your SFT stage used LoRA, first bake it with
    # scripts/merge_lora_ckpt.py so these keys map onto a fresh base.
    if c.get("lora_enabled", False):
        import lora as _lora
        n_w, n_ap, _ = _lora.inject_lora(
            policy, r=c.get("lora_r", 8), alpha=c.get("lora_alpha", 16),
            dropout=c.get("lora_dropout", 0.0),
            targets=tuple(c.get("lora_targets", _lora.DEFAULT_TARGETS)))
        policy.to(device)
        _lora.mark_only_lora_trainable(policy)
        n_all = sum(p.numel() for p in policy.parameters())
        print(f"LoRA: policy wrapped {n_w} layers (r={c.get('lora_r', 8)}, "
              f"alpha={c.get('lora_alpha', 16)}), {n_ap} adapter params "
              f"({100*n_ap/n_all:.2f}%) trainable; reference is the frozen base")

    opts = build_optimizers(policy, c, device)

    # Generation sanity-check (see gen_check() docstring for why this exists).
    # Default ON at ckpt_every cadence; set gen_check_every: 0 to disable.
    gen_check_every = c.get("gen_check_every", c.get("ckpt_every", 2000))
    gen_check_prompt = c.get("gen_check_prompt", "The capital of France is")

    encode, gen_decode = make_encoder_pair(c)
    if gen_check_every > 0:
        print(f"gen-check ENABLED every {gen_check_every} steps, "
              f"prompt={gen_check_prompt!r}")
    max_len = c.get("dpo_max_len", c["block_size"])
    # dpo_jsonl (local generated pairs) and dpo_dataset (HF) can both be set;
    # the pair lists are concatenated. That is how the why-narrated / abstention
    # pairs get blended with UltraFeedback instead of replacing it.
    raw = []
    if c.get("dpo_jsonl"):
        raw += load_preferences_jsonl(c["dpo_jsonl"], c.get("dpo_pair_types"))
    if c.get("dpo_dataset") or not raw:
        ds = c.get("dpo_dataset", "HuggingFaceH4/ultrafeedback_binarized")
        print(f"loading preferences {ds}...")
        hf = load_preferences(ds, c.get("dpo_split", "train_prefs"))
        frac = float(c.get("dpo_dataset_frac", 1.0))
        if frac < 1.0:
            hf = hf[:max(1, int(len(hf) * frac))]
            print(f"  subsampled HF pairs to {len(hf)} ({frac:.0%})")
        raw += hf
    print(f"  raw: {len(raw)} preference pairs")
    examples, skipped = build_dpo_examples(raw, encode, max_len)
    print(f"  encoded: {len(examples)} (skipped {skipped})")
    loader = DPOLoader(examples, c["batch_size"], seed=c.get("seed", 0))

    # Resume from a DPO ckpt in our own run dir if present (else start fresh
    # from the SFT init loaded above).
    ckdir = f"/home/glenn/projects/neuro/models/checkpoints/{c['run_name']}"
    os.makedirs(ckdir, exist_ok=True)
    step, tokens = 0, 0
    cks = sorted(glob.glob(f"{ckdir}/step_*.pt"))
    if cks:
        s = torch.load(cks[-1], map_location=device, weights_only=False)
        policy.load_state_dict(s["model"], strict=False)
        if "opt_states" in s and len(s["opt_states"]) == len(opts):
            for o, st in zip(opts, s["opt_states"]): o.load_state_dict(st)
        step, tokens = s["step"], s["tokens"]
        print(f"resumed DPO ckpt @ step {step}, {tokens/1e6:.1f}M tok")
    per_step_tok = c["batch_size"] * c["grad_accum"] * max_len * 2  # chosen + rejected
    total_steps = max(1, c["max_tokens"] // per_step_tok)
    grad_clip = c.get("grad_clip", 1.0)
    # SimPO uses a higher beta than DPO (paper: ~2.0 for 7B, vs DPO's 0.1).
    # If simpo_beta is set explicitly, use it; else fall back to dpo_beta.
    beta = float(c.get("simpo_beta" if simpo else "dpo_beta",
                        c.get("dpo_beta", 0.1)))

    print(f"{objective.upper()} | params={sum(p.numel() for p in policy.parameters())/1e6:.1f}M "
          f"| beta={beta} | per_step_tok={per_step_tok} | total_steps≈{total_steps}\n")
    t0 = time.time(); win = t0; amp = c["amp"] and device == "cuda"

    while tokens < c["max_tokens"]:
        policy.train()
        for o in opts: o.zero_grad(set_to_none=True)
        loss_val = 0.0; acc = 0.0; spike_val = 0.0
        for _ in range(c["grad_accum"]):
            ids, mask = next(loader)
            ids = ids.to(device); mask = mask.to(device)
            B = ids.shape[0] // 2
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                if simpo:
                    # SimPO: length-normalized avg-logp reward, no reference.
                    avg_pi, sr = response_logp(policy, ids, mask, c["vocab"],
                                                average=True, return_stats=True)
                    pi_w, pi_l = avg_pi[:B], avg_pi[B:]
                    margin = beta * (pi_w - pi_l) - simpo_gamma
                    loss = -F.logsigmoid(margin).mean()
                elif orpo:
                    # ORPO: SFT-NLL(chosen) + lambda * odds-ratio penalty; no ref.
                    # The SFT-NLL term pulls chosen logprob UP in absolute terms,
                    # which is the direct defense against likelihood displacement.
                    avg_pi, sr = response_logp(policy, ids, mask, c["vocab"],
                                                average=True, return_stats=True)
                    lp_w, lp_l = avg_pi[:B], avg_pi[B:]
                    log_odds_w = lp_w - log1mexp(lp_w)
                    log_odds_l = lp_l - log1mexp(lp_l)
                    margin = log_odds_w - log_odds_l
                    l_or = -F.logsigmoid(margin)          # odds-ratio term
                    l_sft = -lp_w                         # NLL of chosen (len-norm)
                    loss = (l_sft + orpo_lambda * l_or).mean()
                else:
                    # Standard DPO: sum-logp with a frozen reference.
                    logp_pi, sr = response_logp(policy, ids, mask, c["vocab"],
                                                 average=False, return_stats=True)
                    with torch.no_grad():
                        logp_ref = response_logp(reference, ids, mask, c["vocab"],
                                                  average=False, return_stats=False)
                    pi_w, pi_l = logp_pi[:B], logp_pi[B:]
                    ref_w, ref_l = logp_ref[:B], logp_ref[B:]
                    margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))
                    loss = -F.logsigmoid(margin).mean()
                acc += (margin > 0).float().mean().item() / c["grad_accum"]
                if sr is not None:
                    # detach: sr may have requires_grad=True; explicit detach
                    # silences the PyTorch warning and doesn't affect training.
                    spike_val += float(sr.detach()) / c["grad_accum"]
            (loss / c["grad_accum"]).backward()
            loss_val += loss.item() / c["grad_accum"]

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        for o in opts: o.step()
        step += 1; tokens += per_step_tok

        if step % c["log_every"] == 0:
            dt = time.time() - win; win = time.time()
            vram = (f" | vram {torch.cuda.memory_allocated()/1024**3:.1f}/"
                    f"{torch.cuda.memory_reserved()/1024**3:.1f} GB"
                    if device == "cuda" else "")
            spike_str = f" | spike {spike_val:.3f}" if spike_val > 0 else ""
            print(f"step {step} | {tokens/1e6:.1f}M tok | loss {loss_val:.3f} "
                  f"| chosen>rejected {acc*100:.0f}%{spike_str} "
                  f"| {c['log_every']*per_step_tok/dt:,.0f} tok/s{vram}", flush=True)
        if step % c["ckpt_every"] == 0:
            save_ckpt(ckdir, policy, opts, step, tokens, c)
        if gen_check_every > 0 and step % gen_check_every == 0:
            sample = gen_check(policy, encode, gen_decode, gen_check_prompt,
                                40, c["block_size"], device)
            print(f"  [gen-check] {sample!r}", flush=True)

    save_ckpt(ckdir, policy, opts, step, tokens, c)
    print(f"done: {tokens/1e6:.1f}M tok in {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    main()
