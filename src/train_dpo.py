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
from train_sft import build_model, build_optimizers, load_lut_encode, save_ckpt


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


def sum_response_logp(model, ids, mask, vocab):
    """Sum of log p(token_t | tokens<t) over positions where mask[t]=True (response tokens)."""
    logits = model(ids)                          # (B2, T, V)
    logp = F.log_softmax(logits, dim=-1)
    # Shift: predict ids[t] from logits at [t-1]
    target = ids[:, 1:]                          # (B2, T-1)
    pred = logp[:, :-1, :]                       # (B2, T-1, V)
    m = mask[:, 1:]                              # (B2, T-1) — mark response positions
    chosen_logp = pred.gather(2, target.unsqueeze(-1)).squeeze(-1)   # (B2, T-1)
    chosen_logp = chosen_logp * m.float()
    return chosen_logp.sum(dim=1)                # (B2,)


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

    # Reference model: frozen copy of policy at SFT init
    reference = build_model(c).to(device)
    reference.load_state_dict(blob["model"], strict=False)
    for p in reference.parameters(): p.requires_grad_(False)
    reference.eval()

    opts = build_optimizers(policy, c, device)

    encode = load_lut_encode(c["vocab"])
    max_len = c.get("dpo_max_len", c["block_size"])
    print(f"loading preferences {c.get('dpo_dataset','HuggingFaceH4/ultrafeedback_binarized')}...")
    raw = load_preferences(c.get("dpo_dataset", "HuggingFaceH4/ultrafeedback_binarized"),
                            c.get("dpo_split", "train_prefs"))
    print(f"  raw: {len(raw)} preference pairs")
    examples, skipped = build_dpo_examples(raw, encode, max_len)
    print(f"  encoded: {len(examples)} (skipped {skipped})")
    loader = DPOLoader(examples, c["batch_size"], seed=c.get("seed", 0))

    ckdir = f"/home/glenn/projects/neuro/models/checkpoints/{c['run_name']}"
    os.makedirs(ckdir, exist_ok=True)
    step, tokens = 0, 0
    per_step_tok = c["batch_size"] * c["grad_accum"] * max_len * 2  # chosen + rejected
    total_steps = max(1, c["max_tokens"] // per_step_tok)
    grad_clip = c.get("grad_clip", 1.0)
    beta = c.get("dpo_beta", 0.1)

    print(f"DPO | params={sum(p.numel() for p in policy.parameters())/1e6:.1f}M "
          f"| beta={beta} | per_step_tok={per_step_tok} | total_steps≈{total_steps}\n")
    t0 = time.time(); win = t0; amp = c["amp"] and device == "cuda"

    while tokens < c["max_tokens"]:
        policy.train()
        for o in opts: o.zero_grad(set_to_none=True)
        loss_val = 0.0; acc = 0.0
        for _ in range(c["grad_accum"]):
            ids, mask = next(loader)
            ids = ids.to(device); mask = mask.to(device)
            B = ids.shape[0] // 2
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                logp_pi = sum_response_logp(policy, ids, mask, c["vocab"])
                with torch.no_grad():
                    logp_ref = sum_response_logp(reference, ids, mask, c["vocab"])
                # split chosen vs rejected
                pi_w, pi_l = logp_pi[:B], logp_pi[B:]
                ref_w, ref_l = logp_ref[:B], logp_ref[B:]
                log_ratios = beta * ((pi_w - ref_w) - (pi_l - ref_l))
                loss = -F.logsigmoid(log_ratios).mean()
                acc += (log_ratios > 0).float().mean().item() / c["grad_accum"]
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
            print(f"step {step} | {tokens/1e6:.1f}M tok | loss {loss_val:.3f} "
                  f"| chosen>rejected {acc*100:.0f}% "
                  f"| {c['log_every']*per_step_tok/dt:,.0f} tok/s{vram}", flush=True)
        if step % c["ckpt_every"] == 0:
            save_ckpt(ckdir, policy, opts, step, tokens, c)

    save_ckpt(ckdir, policy, opts, step, tokens, c)
    print(f"done: {tokens/1e6:.1f}M tok in {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    main()
