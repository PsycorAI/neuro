"""On-policy distillation (OPD) for PsycorNeuro -- post-training stage 3.

    pretrain -> SFT -> ORPO -> **OPD**

The student samples its own trajectories; a frozen teacher then scores EVERY
token of them. That gives the on-policy state distribution of RL with the dense
per-token supervision of distillation, and avoids both RL's sparse reward and
SFT's compounding exposure bias. This is the stage Qwen3 / MiMo / GLM-5 /
MiniCPM5 all converged on in 2026.

Objective (per response token, at student-visited states s_t):

    L_t = KL( pi_s(.|s_t) || pi_t(.|s_t) )          <- REVERSE KL, mode-seeking

We only get the teacher's top-K, so the tractable form is a support-restricted
reverse KL plus an explicit penalty on student mass that falls OUTSIDE the
teacher's support:

    S'   = teacher top-K indices projected to student vocab, index 0 dropped
    p~,q~= student / teacher probs renormalized over S'
    L_t  = sum_{a in S'} p~(a) log(p~(a)/q~(a))  +  lambda_out * -log sum_{a in S'} pi_s(a)

The second term is not a hack: in the true reverse KL the a-not-in-S' region is
where q ~ 0, so it dominates. -log(mass_in_support) is its stable surrogate --
"do not put mass where the teacher has none", which is exactly what reverse KL
buys over forward KL.

Dropping index 0 is exact, not an approximation: vocab_map_32768 is INJECTIVE on
its nonzero values (verified: 32767 nonzero entries, 32767 distinct), so the only
way the projected top-K can contain duplicates is via id 0, which absorbs all
74.5% of the Llama-3 vocab that is outside our student vocab. Index 0 is also
banned during sampling, so it never appears in a trajectory either.

Teacher requirement: it MUST use the Llama-3 tokenizer. Our student vocab is a
LUT projection of Llama-3's 128256, so student token i maps back to Llama-3 token
inv[i] and the two sequences align position-for-position. A teacher with a
different tokenizer (e.g. the cached Phi-3.5) cannot be used here at all.

  CUDA_VISIBLE_DEVICES=1 python src/train_opd.py --config configs/opd_smoke_363m.yaml

Config keys (beyond the standard arch/optim block):
  opd_init          student ckpt run_name or path (mandatory; the ORPO output)
  opd_teacher       HF id of the Llama-3-tokenizer teacher
  opd_prompts       JSONL with an "instruction" field, or an HF dataset id
  opd_max_new       response tokens to sample per rollout (default 96)
  opd_rollouts      rollouts per prompt (default 4, per the recipe paper)
  opd_temperature   sampling temperature (default 1.0, per the recipe paper)
  opd_topk          teacher top-K support (default 16, per the recipe paper)
  opd_lambda_out    weight on the out-of-support mass penalty (default 1.0)
  opd_gen_batch     rollouts generated per generation pass (VRAM lever)
"""
import os, sys, time, math, argparse, glob, json
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from train_sft import (build_model, build_optimizers, load_lut_encode,
                       load_lut_decode, gen_check, save_ckpt,
                       make_encoder_pair)
import kd

CKPT_ROOT = "/home/glenn/projects/neuro/models/checkpoints"
EOS_MARK = "\n###"


# ------------------------------------------------------------------ vocab ---
def build_inv_lut(vocab, device):
    """student id -> Llama-3 id (the inverse of the projection LUT). Student id
    0 (UNK) has no preimage; it is banned during sampling and dropped from the
    teacher support, so its slot is filled with a harmless id."""
    lut = np.load(f"/home/glenn/projects/neuro/data/vocab_map_{vocab}.npy")
    inv = np.zeros(vocab, dtype=np.int64)
    kept = np.where(lut > 0)[0]
    inv[lut[kept]] = kept
    return (torch.from_numpy(inv).to(device),
            torch.from_numpy(lut).long().to(device))


# ------------------------------------------------------------- generation ---
@torch.no_grad()
def batched_rollout(model, prompt_ids, max_new, temperature, block_size,
                    device, eos_ids, ban_zero=True):
    """Sample continuations for a batch of variable-length prompts.

    The model takes no attention mask, so padding has to be handled by
    construction rather than by masking: sequences are RIGHT-padded and each
    one's logits are read at its OWN last real position. Because the model is
    causal, position L-1 cannot see the pad tokens that sit after it, so this is
    exact -- unlike left-padding, which would feed pads into the Hebbian state
    ahead of the real prompt.

    Full-prefix re-forward per token (not incremental initial_state decoding):
    the hybrid attention layer keeps NO KV cache (src/hybrid_attn.py), so
    single-token stepping would show it a length-1 sequence and produce logits
    that differ from the loss-time forward. Re-forwarding keeps the rollouts
    genuinely on-policy w.r.t. the distribution the loss differentiates.

    Returns (seq (B,Tmax), lens (B,), resp_start (B,)).
    """
    B = len(prompt_ids)
    starts = torch.tensor([len(p) for p in prompt_ids], device=device)
    total = int(starts.max()) + max_new
    seq = torch.zeros(B, total, dtype=torch.long, device=device)
    for i, p in enumerate(prompt_ids):
        seq[i, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
    lens = starts.clone()
    done = torch.zeros(B, dtype=torch.bool, device=device)
    ar = torch.arange(B, device=device)
    n_eos = len(eos_ids)

    for _ in range(max_new):
        window = int(lens.max())
        lo = max(0, window - block_size)
        logits = model(seq[:, lo:window])                    # (B, window-lo, V)
        idx = (lens - 1 - lo).clamp(min=0)
        last = logits[ar, idx].float() / max(temperature, 1e-6)
        if ban_zero:
            last[:, 0] = float("-inf")                       # UNK has no teacher preimage
        nxt = torch.multinomial(F.softmax(last, dim=-1), 1).squeeze(1)
        nxt = torch.where(done, torch.zeros_like(nxt), nxt)
        seq[ar, lens] = nxt
        lens = lens + (~done).long()
        if n_eos:                                            # stop on the EOS token run
            hit = torch.ones(B, dtype=torch.bool, device=device)
            for k, e in enumerate(eos_ids):
                pos = lens - n_eos + k
                ok = (pos >= 0) & (seq[ar, pos.clamp(min=0)] == e)
                hit &= ok
            done |= hit
        if bool(done.all()):
            break
    return seq[:, :int(lens.max())], lens, starts


# ------------------------------------------------------------------- loss ---
def opd_loss(student_logits, teacher_topk, resp_mask, lambda_out=1.0):
    """Support-restricted reverse KL + out-of-support mass penalty.

    student_logits: (B, T, V) -- logits that PREDICT position t (already shifted
                    by the caller so index t lines up with teacher_topk t)
    teacher_topk:   (probs (B,T,K), student_idx (B,T,K)) from kd.teacher_distribution
    resp_mask:      (B, T) bool -- response positions only
    Returns (loss, stats dict).
    """
    q_raw, idx = teacher_topk
    idx = idx.long()
    # Drop index 0 (the UNK sink that absorbs every out-of-student-vocab teacher
    # token). Everything else is distinct because the LUT is injective, so after
    # this the support has no duplicates and the renormalization below is exact.
    valid = idx != 0
    # A position whose ENTIRE teacher top-K lands outside our student vocab
    # carries no usable signal. It has to be dropped, not merely renormalized:
    # renormalizing an all-invalid row divides by zero and NaNs the whole step.
    has_sup = valid.any(-1)                                      # (B,T)
    q = q_raw.float() * valid
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-9)

    logp = F.log_softmax(student_logits.float(), dim=-1)          # (B,T,V)
    lp_s = logp.gather(-1, idx)                                  # (B,T,K)
    # student mass inside the teacher's support (log-space, duplicate-free)
    lp_in = torch.logsumexp(lp_s.masked_fill(~valid, float("-inf")), dim=-1)
    lp_in = torch.where(has_sup, lp_in, torch.zeros_like(lp_in))
    # renormalized student distribution ON the support
    p = (lp_s - lp_in.unsqueeze(-1)).exp() * valid

    rkl = (p * ((p.clamp_min(1e-9)).log() - q.clamp_min(1e-9).log())).sum(-1)
    out_pen = -lp_in                                             # -log mass_in_support
    per_tok = rkl + lambda_out * out_pen

    m = resp_mask.float() * has_sup.float()
    denom = m.sum().clamp_min(1.0)
    loss = (per_tok * m).sum() / denom
    with torch.no_grad():
        r = resp_mask.float().sum().clamp_min(1.0)
        stats = {
            "rkl": float((rkl * m).sum() / denom),
            "mass_in": float((lp_in.exp() * m).sum() / denom),
            "H_s": float((-(logp.exp() * logp).sum(-1) * m).sum() / denom),
            # fraction of response positions with no usable teacher support --
            # if this is not ~0 the teacher is speaking vocabulary we lack.
            "no_sup": float(((~has_sup).float() * resp_mask.float()).sum() / r),
        }
    return loss, stats


# ------------------------------------------------------------------ data ----
def load_prompts(c):
    """Prompt pool for the rollouts. Prefer the SAME prompt distribution the
    teacher was post-trained on (the recipe paper's 'teacher-aligned prompts'),
    while keeping some breadth so entropy does not collapse."""
    src = c.get("opd_prompts")
    out = []
    if src and str(src).endswith(".jsonl"):
        with open(src, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                p = r.get("instruction") or r.get("prompt")
                if p and 20 < len(p) < 600:
                    out.append(p.strip())
    else:
        from datasets import load_dataset
        ds = load_dataset(src or "yahma/alpaca-cleaned", split="train")
        for ex in ds:
            inp = (ex.get("input") or "").strip()
            p = (ex.get("instruction") or "").strip()
            if inp:
                p = p + "\n\n" + inp
            if 20 < len(p) < 600:
                out.append(p)
    rng = np.random.default_rng(c.get("seed", 0))
    rng.shuffle(out)
    return out


PROMPT_FMT = "### Instruction:\n{message}\n\n### Response:\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    with open(args.config) as f:
        c = yaml.safe_load(f)
    device = args.device
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(c.get("seed", 0))

    vocab = c["vocab"]
    max_new = int(c.get("opd_max_new", 96))
    rollouts = int(c.get("opd_rollouts", 4))
    temperature = float(c.get("opd_temperature", 1.0))
    topk = int(c.get("opd_topk", 16))
    lambda_out = float(c.get("opd_lambda_out", 1.0))
    gen_batch = int(c.get("opd_gen_batch", c["batch_size"] * 2))
    max_len = c.get("opd_max_len", c["block_size"])

    # ---- student ----------------------------------------------------------
    student = build_model(c).to(device)
    init = c["opd_init"]
    if "/" not in init:
        cks = sorted(glob.glob(f"{CKPT_ROOT}/{init}/step_*.pt"))
        if not cks:
            raise FileNotFoundError(f"no ckpt for opd_init={init}")
        init = cks[-1]
    blob = torch.load(init, map_location=device, weights_only=False)
    miss, unexp = student.load_state_dict(blob["model"], strict=False)
    if unexp:
        print(f"  [info] ignoring {len(unexp)} ckpt keys (e.g. {unexp[0]})")
    if miss:
        print(f"  [warn] {len(miss)} keys missing in ckpt: {miss[:3]}")
    print(f"loaded student init: {init}")

    opts = build_optimizers(student, c, device)
    # Base LRs from the FRESH optimizers, before a resume loads optimizer state: that state holds the already-scheduled
    # lr, and reading it as the base multiplied the LR by lr_scale(save step) on every resume (train_gpu, 2026-10-10).
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]

    # ---- teacher ----------------------------------------------------------
    teacher_id = c["opd_teacher"]
    t_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    teacher = kd.load_teacher(teacher_id, device=device, dtype=t_dtype,
                              compile=bool(c.get("opd_teacher_compile", False)))
    tk_vocab = teacher.get_input_embeddings().weight.shape[0]
    if tk_vocab < 128000:
        raise RuntimeError(
            f"teacher {teacher_id} has vocab {tk_vocab}; OPD needs a Llama-3 "
            f"tokenizer (128256) so student tokens map back 1:1 via the LUT")
    print(f"teacher: {teacher_id} (vocab {tk_vocab})")

    inv_t, lut_t = build_inv_lut(vocab, device)

    # ---- data -------------------------------------------------------------
    encode, decode = make_encoder_pair(c)
    eos_ids = encode(EOS_MARK)
    print(f"EOS_MARK {EOS_MARK!r} -> ids {eos_ids}")
    prompts = load_prompts(c)
    print(f"prompt pool: {len(prompts)}")
    enc_prompts = []
    for p in prompts:
        ids = encode(PROMPT_FMT.format(message=p))
        if 8 < len(ids) < max_len - max_new:
            enc_prompts.append(ids)
    print(f"  usable encoded prompts: {len(enc_prompts)}")
    if not enc_prompts:
        raise RuntimeError("no usable prompts after encoding")

    # ---- resume -----------------------------------------------------------
    ckdir = f"{CKPT_ROOT}/{c['run_name']}"
    os.makedirs(ckdir, exist_ok=True)
    step, tokens = 0, 0
    cks = sorted(glob.glob(f"{ckdir}/step_*.pt"))
    if cks:
        s = torch.load(cks[-1], map_location=device, weights_only=False)
        student.load_state_dict(s["model"], strict=False)
        if "opt_states" in s and len(s["opt_states"]) == len(opts):
            for o, st in zip(opts, s["opt_states"]):
                o.load_state_dict(st)
        step, tokens = s["step"], s["tokens"]
        print(f"resumed OPD ckpt @ step {step}, {tokens/1e6:.1f}M tok")

    per_step_tok = c["batch_size"] * c["grad_accum"] * max_new
    total_steps = max(1, c["max_tokens"] // per_step_tok)
    warmup = int(c.get("lr_warmup_frac", 0.03) * total_steps)
    lr_min_frac = c.get("lr_min_frac", 0.1)
    grad_clip = c.get("grad_clip", 1.0)

    def lr_scale(s):
        if s < warmup:
            return s / max(1, warmup)
        pr = (s - warmup) / max(1, total_steps - warmup)
        return lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))

    gen_check_every = c.get("gen_check_every", c.get("ckpt_every", 500))
    gen_check_prompt = c.get("gen_check_prompt",
                             PROMPT_FMT.format(message="What is the capital of France?"))
    amp = c["amp"] and device == "cuda"
    rng = np.random.default_rng(c.get("seed", 0) + step)
    nonfinite = 0

    print(f"OPD | student {sum(p.numel() for p in student.parameters())/1e6:.1f}M "
          f"| max_new={max_new} rollouts={rollouts} T={temperature} topk={topk} "
          f"lambda_out={lambda_out}\n| per_step_tok={per_step_tok} "
          f"total_steps~{total_steps} warmup={warmup}\n", flush=True)

    t0 = time.time(); win = t0
    while tokens < c["max_tokens"]:
        scale = lr_scale(step)
        for o, lrs in zip(opts, base_lrs):
            for g, base in zip(o.param_groups, lrs):
                g["lr"] = base * scale
        for o in opts:
            o.zero_grad(set_to_none=True)
        agg = {"loss": 0.0, "rkl": 0.0, "mass_in": 0.0, "H_s": 0.0, "no_sup": 0.0}

        for _ in range(c["grad_accum"]):
            # ---- 1. on-policy rollouts (no grad) -------------------------
            n_prompts = max(1, c["batch_size"] // rollouts)
            pick = rng.integers(0, len(enc_prompts), size=n_prompts)
            batch_prompts = []
            for i in pick:
                batch_prompts.extend([enc_prompts[i]] * rollouts)
            student.eval()
            seq, lens, starts = batched_rollout(
                student, batch_prompts, max_new, temperature,
                c["block_size"], device, eos_ids)

            # response-position mask: True where a SAMPLED token sits
            T = seq.shape[1]
            pos = torch.arange(T, device=device).unsqueeze(0)
            resp = (pos >= starts.unsqueeze(1)) & (pos < lens.unsqueeze(1))

            # ---- 2. teacher scores every token of them -------------------
            raw = inv_t[seq]                         # student ids -> Llama-3 ids
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                t_topk = kd.teacher_distribution(teacher, raw, lut_t,
                                                 student_vocab=vocab, topk=topk)

            # ---- 3. reverse KL at the visited states ---------------------
            student.train()
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                logits = student(seq)
            # logits[:, t-1] predicts token t -> align both sides to "position t"
            loss, st = opd_loss(logits[:, :-1], (t_topk[0][:, 1:], t_topk[1][:, 1:]),
                                resp[:, 1:], lambda_out=lambda_out)
            if not torch.isfinite(loss):
                nonfinite += 1
                print(f"  [skip] non-finite loss at step {step} (total {nonfinite})",
                      flush=True)
                continue
            (loss / c["grad_accum"]).backward()
            agg["loss"] += float(loss.detach()) / c["grad_accum"]
            for k in ("rkl", "mass_in", "H_s", "no_sup"):
                agg[k] += st[k] / c["grad_accum"]

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        for o in opts:
            o.step()
        step += 1
        tokens += per_step_tok

        if step % c["log_every"] == 0:
            dt = time.time() - win; win = time.time()
            vram = (f" | vram {torch.cuda.memory_allocated()/1024**3:.1f}/"
                    f"{torch.cuda.memory_reserved()/1024**3:.1f} GB"
                    if device == "cuda" else "")
            print(f"step {step} | {tokens/1e6:.2f}M tok | loss {agg['loss']:.4f} "
                  f"| rkl {agg['rkl']:.4f} | mass_in {agg['mass_in']:.3f} "
                  f"| H_s {agg['H_s']:.3f} | no_sup {agg['no_sup']:.3f} "
                  f"| {c['log_every']*per_step_tok/dt:,.0f} tok/s{vram}", flush=True)
        if step % c["ckpt_every"] == 0:
            save_ckpt(ckdir, student, opts, step, tokens, c)
        if gen_check_every > 0 and step % gen_check_every == 0:
            sample = gen_check(student, encode, decode, gen_check_prompt,
                               48, c["block_size"], device)
            print(f"  [gen-check] {sample!r}", flush=True)

    save_ckpt(ckdir, student, opts, step, tokens, c)
    print(f"done: {tokens/1e6:.1f}M tok in {(time.time()-t0)/3600:.2f}h "
          f"({nonfinite} skipped microsteps)")


if __name__ == "__main__":
    main()
