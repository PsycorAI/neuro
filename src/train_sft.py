"""SFT (supervised fine-tuning) trainer for PsycorNeuro.

Loads a pretrained base checkpoint, trains on instruction-response pairs
(Alpaca-format), with loss masked to response tokens only. Saves to a new
run_name. Resume from the SFT ckpt works too (auto-resume from latest step_*).

  python src/train_sft.py --config configs/sft_30m.yaml

Config keys (in addition to standard arch/optim):
  sft_init:     base ckpt path or run_name (mandatory)
  sft_dataset:  HF dataset id (default: yahma/alpaca-cleaned)
  sft_split:    dataset split (default: train)
  sft_max_len:  max sequence length for SFT (default: block_size)
"""
import os, sys, time, math, argparse, glob, signal
import numpy as np
import torch
import torch.nn.functional as F
import yaml
sys.path.insert(0, os.path.dirname(__file__))
from sft_data import load_alpaca, build_sft_examples, SFTLoader


def build_model(c):
    # Delegate to the maintained builder in train_gpu so SFT never drifts from
    # the pretrain architecture. This function previously enumerated its own
    # (much shorter) kwarg list and silently omitted gdn2/engram_v2/mtp/rmsnorm/
    # embed_scale/sdm flags — building a 44M model that could not load a 57M
    # current-recipe base (the base's gdn2/engram/mtp weights were dropped as
    # "unexpected" and the norm silently fell back to LayerNorm). Delegating
    # keeps a single source of truth for arch construction. Lazy import: keeps
    # train_sft's module-load surface small and avoids any import ordering risk.
    from train_gpu import build_model as _build_model
    return _build_model(c)


def build_optimizers(model, c, device):
    # LoRA: adapters (lora_A/lora_B) are 2D but must NOT go through Muon —
    # Newton-Schulz orthogonalization is the wrong update rule for low-rank
    # factors, and with only adapters trainable the Muon path would hand AdamW
    # an empty param list. Train adapters with a single fused AdamW.
    if c.get("lora_enabled", False):
        trainable = [p for p in model.parameters() if p.requires_grad]
        fused_kw = {"fused": True} if device == "cuda" else {}
        opt = torch.optim.AdamW(trainable, lr=c["lr"], **fused_kw)
        print(f"optimizer: AdamW (LoRA adapters only, "
              f"{sum(p.numel() for p in trainable)} params, lr={c['lr']})")
        return [opt]
    # Delegate to the maintained builder in train_gpu so the SFT optimizer never
    # drifts from the pretrain recipe. The previous local copy passed only
    # muon_lr/adamw_lr/adamw_fused and SILENTLY DROPPED normuon(+beta2/eps),
    # fix_gate_matrices, precise_gate_filter, route_embeddings_by_type,
    # weight_decay and adamw_8bit. Caught live on the 363M flagship SFT launch:
    # pretrain built NorMuon(71 matrices)+AdamW(123), SFT built Muon(80)+AdamW(114)
    # -- 9 one-row gate matrices (one per layer) were going through Muon's
    # Newton-Schulz, which degenerates on 1-row matrices and kills beta (memory
    # neuro_muon_gate_matrix_bug). Same drift class as the build_model fix above.
    from train_gpu import _build_optimizers as _canonical_opts
    return _canonical_opts(model, c, device)


def load_lut_encode(vocab):
    """Returns text→list[int] encoder using the project LUT + Llama-3 tokenizer."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
    lut_path = f"/home/glenn/projects/neuro/data/vocab_map_{vocab}.npy"
    lut = np.load(lut_path)
    def encode(text):
        ids = tk.encode(text, add_special_tokens=False)
        return [int(lut[t]) for t in ids if lut[t] != 0]
    return encode


def load_lut_decode(vocab):
    """Returns list[int]->text decoder, the inverse of load_lut_encode. Used by
    the in-training generation sanity-check (see train_dpo.py gen_check_every),
    added 2026-07-26 after a SimPO run collapsed to incoherent word-salad
    output undetected for ~2h of GPU time -- loss/accuracy metrics alone did
    not catch it. See neuro_simpo_collapse_2026-07-26 memory."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
    lut_path = f"/home/glenn/projects/neuro/data/vocab_map_{vocab}.npy"
    lut = np.load(lut_path)
    inv = np.full(vocab, -1, dtype=np.int64)
    nz = np.where(lut > 0)[0]
    inv[lut[nz]] = nz
    def decode(ids):
        return tk.decode([int(inv[i]) for i in ids if 0 < i < vocab and inv[i] >= 0])
    return decode


def make_encoder_pair(c):
    """(encode, decode) for this run.

    lut_subword_fallback: true routes through src/lut_spell, which re-spells
    Llama-3 tokens that are outside our 32768 student vocab out of in-vocab
    subwords instead of DROPPING them. Required for any post-training data with
    vocabulary outside the pretraining head -- the plain encoder turns
    'calculator' into '' and 'kilometers' into 'ometers', which silently
    corrupted the tool-call targets (see src/lut_spell docstring).

    Default false = byte-identical to the original behaviour, so existing runs
    and their checkpoints reproduce exactly.
    """
    if c.get("lut_subword_fallback", False):
        import lut_spell
        enc = lut_spell.make_encoder(c["vocab"], subword_fallback=True, report=True)
        print("[encoder] subword-fallback ENABLED (src/lut_spell)")
        return enc, lut_spell.make_decoder(c["vocab"])
    return load_lut_encode(c["vocab"]), load_lut_decode(c["vocab"])


def load_sft_corpus(c, encode, max_len):
    """Build the encoded (input_ids, labels) list for this SFT run.

    Two sources, selected by config; both can be set and are concatenated:
      sft_jsonl: <path|list>  -- local JSONL from scripts/gen_teacher_data.py.
          The record shape is auto-detected per file so one key covers both new
          post-training stages:
            * has "target_mode" -> tool records, formatted by src/tool_data
              (### Tools: block + TOOL/ANSWER response)          [idea #4]
            * {instruction, output} -> the ordinary Alpaca path, which is
              exactly what the OPD off-policy cold start is      [idea #1]
      sft_dataset: <hf id>    -- unchanged original behaviour (the default).

    Mixing matters for the tool stage: training on tool records alone teaches a
    model to emit nothing but tool calls, so the general instruction corpus is
    kept in the blend (sft_dataset_frac controls how much).
    """
    import json as _json
    parts = []
    paths = c.get("sft_jsonl")
    if paths:
        if isinstance(paths, str):
            paths = [paths]
        import tool_data
        tools = tool_data.load_tools()
        for p in paths:
            head = None
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        head = _json.loads(line)
                        break
            if head is None:
                print(f"  [warn] {p} is empty -- skipping")
                continue
            if "target_mode" in head:
                # split="train": the hash-held-out test split is what tool_eval/head_eval score on
                pairs, bad = tool_data.load_tool_jsonl(p, tools, split="train", proto=c.get("tool_proto", 1),
                                                       full_frac=c.get("sft_tools_full_frac", 0.0))
                enc, skipped = tool_data.build_tool_examples(pairs, encode, max_len)
                kinds = {}
                for q in pairs:
                    kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
                print(f"  {p}: {len(pairs)} tool pairs {kinds} -> encoded "
                      f"{len(enc)} (bad {bad}, skipped {skipped})")
            else:
                raw = []
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue    # partial trailing line from a killed writer
                        if r.get("instruction") and r.get("output"):
                            raw.append({"instruction": r["instruction"],
                                        "input": r.get("input", ""),
                                        "output": r["output"],
                                        "history": r.get("history")})   # 10-21: chat brain (sft_data.format_example)
                enc, skipped = build_sft_examples(raw, encode, max_len)
                print(f"  {p}: {len(raw)} instruction pairs -> encoded "
                      f"{len(enc)} (skipped {skipped})")
            parts.append(enc)
    if c.get("sft_dataset") or not parts:
        ds = c.get("sft_dataset", "yahma/alpaca-cleaned")
        print(f"loading SFT dataset {ds}...")
        raw = load_alpaca(ds, c.get("sft_split", "train"))
        frac = float(c.get("sft_dataset_frac", 1.0))
        if frac < 1.0:
            keep = max(1, int(len(raw) * frac))
            raw = raw[:keep]
            print(f"  subsampled to {keep} ({frac:.0%})")
        print(f"  raw: {len(raw)} examples")
        enc, skipped = build_sft_examples(raw, encode, max_len)
        print(f"  encoded: {len(enc)} (skipped {skipped})")
        parts.append(enc)
    out = [e for part in parts for e in part]
    # sft_text_frac (2026-10-14, demo plan D8; default 0 = unchanged): that share of the examples is plain pretraining
    # text, windows of sft_max_len tokens from the training stream (data_bin / data_prelut), loss on every token. Aimed
    # at the raw-text perplexity loss of SFT (13L rehearsal: 36.55 -> 54.59). Fixed seed, drawn once.
    tf = float(c.get("sft_text_frac", 0.0) or 0.0)
    if tf > 0:
        import numpy as _np
        from data_stream import StreamText
        st = StreamText(c["data_bin"], vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                        val_tokens=c.get("data_val_tokens"), prelut=c.get("data_prelut", False))
        n_text = int(round(len(out) * tf / (1.0 - tf)))
        _np.random.seed(int(c.get("seed", 0)) + 7)
        txt = []
        while len(txt) < n_text:
            x, _ = st.batch(min(256, n_text - len(txt)), max_len, "train", "cpu")
            txt += [(r.tolist(), r.tolist()) for r in x]
        out = out + txt
        print(f"  + {len(txt)} plain-text windows (sft_text_frac {tf}, {max_len} tok each)")
    print(f"  TOTAL encoded examples: {len(out)}")
    return out


@torch.no_grad()
def gen_check(model, encode, decode, prompt, n_new, block_size, device):
    """Cheap in-training generation sanity-check. Greedy decode (no sampling
    randomness) so successive calls are directly comparable step-over-step.
    Shared by train_sft.py and train_dpo.py.

    Added 2026-07-26 after a SimPO run (beta=2.0, reference-free) collapsed to
    incoherent word-salad / repetition-loop output by ~37% through its token
    budget, undetected for ~2h of GPU time because loss and chosen>rejected
    accuracy both looked fine (in fact suspiciously "too good": near-zero loss,
    100% accuracy -- the actual signature of the collapse, not evidence against
    it). Printing real generated text periodically is the one check that would
    have caught this immediately. See neuro_simpo_collapse_2026-07-26 memory.
    """
    was_training = model.training
    model.eval()
    ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    if ids.shape[1] == 0:
        return "(prompt encoded to empty -- skipping)"
    out = ids
    for _ in range(n_new):
        x = out[:, -block_size:]
        logits = model(x)[0, -1]
        nxt = logits.argmax(dim=-1, keepdim=True)
        out = torch.cat([out, nxt.view(1, 1)], dim=1)
    if was_training:
        model.train()
    return decode(out[0].tolist())


def save_ckpt(ckdir, model, opts, step, tokens, c):
    p = f"{ckdir}/step_{step:07d}.pt"
    torch.save({"model": model.state_dict(),
                "opt_states": [o.state_dict() for o in opts],
                "step": step, "tokens": tokens, "cfg": c}, p)
    for o in sorted(glob.glob(f"{ckdir}/step_*.pt"))[:-c.get("keep_last", 2)]:
        os.remove(o)
    print(f"  saved {p}")
    # LoRA: also emit the tiny adapter-only artifact (the swappable skill mod),
    # independent of the base checkpoint. Records the base it was trained on so
    # it can be re-applied to that frozen checkpoint later.
    if c.get("lora_enabled", False):
        import lora as _lora
        ad = _lora.lora_state_dict(model)
        ap = f"{ckdir}/adapter_step_{step:07d}.pt"
        torch.save({"lora": ad, "base": c.get("sft_init"), "step": step,
                    "tokens": tokens,
                    "lora_r": c.get("lora_r", 8), "lora_alpha": c.get("lora_alpha", 16),
                    "lora_targets": list(c.get("lora_targets", _lora.DEFAULT_TARGETS))}, ap)
        for o in sorted(glob.glob(f"{ckdir}/adapter_step_*.pt"))[:-c.get("keep_last", 2)]:
            os.remove(o)


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

    # Build model
    model = build_model(c).to(device)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6

    # Load BASE pretrained weights (sft_init) — model-only, optimizer fresh.
    sft_init = c["sft_init"]
    if "/" not in sft_init:
        cks = sorted(glob.glob(
            f"/home/glenn/projects/neuro/models/checkpoints/{sft_init}/step_*.pt"))
        if not cks:
            raise FileNotFoundError(f"no base ckpt found for sft_init={sft_init}")
        sft_init = cks[-1]
    blob = torch.load(sft_init, map_location=device, weights_only=False)
    miss, unexp = model.load_state_dict(blob["model"], strict=False)
    if unexp: print(f"  [info] ignoring {len(unexp)} ckpt keys (e.g. {unexp[0]})")
    if miss:  print(f"  [warn] {len(miss)} keys missing in ckpt: {miss[:3]}")
    # 2026-10-14 (Neuro 61): a config / base mismatch used to train on silently (a dense base under a spike-code config
    # = 74 thetas at init; a converted base under a dense config drops them). Now an error unless explicitly allowed.
    if (miss or unexp) and not c.get("sft_allow_key_mismatch", False):
        raise RuntimeError(f"base checkpoint and config disagree: {len(miss)} missing (e.g. {miss[:2]}), {len(unexp)} "
                           f"unexpected (e.g. {unexp[:2]}); build the config from the base's architecture "
                           f"(spike_code etc.) or set sft_allow_key_mismatch: true")
    print(f"loaded BASE: {sft_init} ({blob.get('tokens',0)/1e9:.2f}B tok)")
    # 2026-10-11: the base ckpt (weights + its optimizer state, ~10 GB at 1B) was loaded onto the device and stayed
    # referenced for the whole run (same leak as train_gpu's resume, found in the 1B smoke). Release it.
    del blob
    if device == "cuda":
        torch.cuda.empty_cache()

    # LoRA (parameter-efficient SFT): inject adapters onto the frozen base AFTER
    # loading pretrained weights, so only the tiny A/B factors train. Default-off;
    # a plain full-finetune run is byte-identical to before.
    lora_enabled = c.get("lora_enabled", False)
    if lora_enabled:
        import lora as _lora
        n_w, n_ap, names = _lora.inject_lora(
            model, r=c.get("lora_r", 8), alpha=c.get("lora_alpha", 16),
            dropout=c.get("lora_dropout", 0.0),
            targets=tuple(c.get("lora_targets", _lora.DEFAULT_TARGETS)))
        model.to(device)
        _lora.mark_only_lora_trainable(model)
        # lora_base_bf16 (2026-10-22, Neuro 73; default false = unchanged): store the FROZEN base weights of the wrapped
        # linears in bf16. Under amp autocast casts them to bf16 at every matmul anyway (same rounding), so the math is
        # the same; it saves ~2.2 GB at 1B (fp32 storage + the per-step bf16 copies) -> a larger micro-batch fits the
        # 5080. Needs amp: every forward must run under autocast (gen_check is wrapped below).
        if c.get("lora_base_bf16", False):
            assert c["amp"] and device == "cuda", "lora_base_bf16 needs amp on cuda"
            for mod in model.modules():
                if isinstance(mod, _lora.LoRALinear):
                    mod.base.to(torch.bfloat16)
            print("LoRA: frozen base linears stored in bf16 (lora_base_bf16)")
        n_all = sum(p.numel() for p in model.parameters())
        print(f"LoRA: wrapped {n_w} layers (r={c.get('lora_r', 8)}, "
              f"alpha={c.get('lora_alpha', 16)}), {n_ap} adapter params "
              f"({100*n_ap/n_all:.2f}% of {n_all/1e6:.1f}M) trainable")

    opts = build_optimizers(model, c, device)
    # Base LRs from the FRESH optimizers, before a resume loads optimizer state: that state holds the already-scheduled
    # lr, and reading it as the base multiplied the LR by lr_scale(save step) on every resume (train_gpu, 2026-10-10).
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]

    # Generation sanity-check (see gen_check() docstring). Default ON at
    # ckpt_every cadence; set gen_check_every: 0 to disable.
    gen_check_every = c.get("gen_check_every", c.get("ckpt_every", 2000))
    gen_check_prompt = c.get("gen_check_prompt", "The capital of France is")

    # Data
    encode, gen_decode = make_encoder_pair(c)
    if gen_check_every > 0:
        print(f"gen-check ENABLED every {gen_check_every} steps, "
              f"prompt={gen_check_prompt!r}")
    max_len = c.get("sft_max_len", c["block_size"])
    examples = load_sft_corpus(c, encode, max_len)
    if not examples:
        raise RuntimeError("SFT corpus is empty -- check sft_jsonl / sft_dataset")

    loader = SFTLoader(examples, c["batch_size"], seed=c.get("seed", 0))

    # Resume from SFT ckpt if present
    ckdir = f"/home/glenn/projects/neuro/models/checkpoints/{c['run_name']}"
    os.makedirs(ckdir, exist_ok=True)
    step, tokens = 0, 0
    cks = sorted(glob.glob(f"{ckdir}/step_*.pt"))
    if cks:
        s = torch.load(cks[-1], map_location=device, weights_only=False)
        model.load_state_dict(s["model"], strict=False)
        if "opt_states" in s and len(s["opt_states"]) == len(opts):
            for o, st in zip(opts, s["opt_states"]): o.load_state_dict(st)
        step, tokens = s["step"], s["tokens"]
        print(f"resumed SFT ckpt @ step {step}, {tokens/1e6:.1f}M tok")
        del s                                   # 2026-10-11: free the ckpt held on the device (see `del blob` above)
        if device == "cuda":
            torch.cuda.empty_cache()

    per_step_tok = c["batch_size"] * c["grad_accum"] * max_len
    total_steps = max(1, c["max_tokens"] // per_step_tok)
    warmup_steps = int(c.get("lr_warmup_frac", 0.02) * total_steps)
    lr_min_frac = c.get("lr_min_frac", 0.1)
    grad_clip = c.get("grad_clip", 1.0)

    def lr_scale(s):
        if s < warmup_steps: return s / max(1, warmup_steps)
        p = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    print(f"arch={c['arch']} params={nparams:.2f}M device={device} | "
          f"per_step_tok={per_step_tok} | total_steps≈{total_steps} | warmup={warmup_steps}\n")
    # sft_xchunk_frac (2026-10-15, Neuro 63; default 0 = unchanged, no RNG drawn): with this probability per micro-batch
    # one extra cross-chunk recall batch (train_gpu.mqar_gap_batch, gap = attn_chunk - 2N + randint(attn_chunk): kv block
    # in attention chunk A, query in chunk B, so only the memory can answer) at mqar_gap_weight, own backward (peak
    # memory = max of the two). SFT examples are sft_max_len (512) < attn_chunk, so without it SFT never exercises
    # beyond-the-window recall (B2: cross-chunk recall 100% -> 40-84% by 30M tokens).
    xc_frac = float(c.get("sft_xchunk_frac", 0) or 0)
    if xc_frac > 0:
        from train_gpu import mqar_gap_batch
        from data_stream import StreamText
        xc_data = StreamText(c["data_bin"], vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                             val_tokens=c.get("data_val_tokens"), prelut=c.get("data_prelut", False))
        xc_rng = np.random.RandomState(int(c.get("seed", 0)) + 11)
        # n74 (2026-10-23): mqar_xchunk_C first (v4f: no attention chunks), as train_gpu's mqar_xchunk; absent = attn_chunk
        xc_C, xc_B = int(c.get("mqar_xchunk_C") or c["attn_chunk"]), int(c.get("sft_xchunk_batch", 2))
        xc_w = float(c.get("mqar_gap_weight", c.get("mqar_weight", 0.1)))
        print(f"  sft_xchunk_frac {xc_frac}: cross-chunk recall batches B{xc_B} x <= {2 * xc_C} tok at weight {xc_w}")

    t0 = time.time(); win = t0
    amp = c["amp"] and device == "cuda"
    # 2026-10-14 (Neuro 61): SIGUSR1 = preempt: save a checkpoint after the current step and exit 3; a relaunch
    # auto-resumes from it (the loader restarts from its seed). Lets a health check borrow the GPU. SIGTERM unchanged.
    preempt = {"hit": False}
    signal.signal(signal.SIGUSR1, lambda *_: preempt.__setitem__("hit", True))
    # sft_stop_tokens (2026-10-15, Neuro 63; default None = max_tokens, unchanged): stop (save + 'done:') after this many
    # tokens while the LR schedule stays the max_tokens one -- an A/B arm equal to another run's first N tokens.
    stop_tokens = int(c.get("sft_stop_tokens") or c["max_tokens"])
    while tokens < stop_tokens:
        model.train()
        scale = lr_scale(step)
        for o, lrs in zip(opts, base_lrs):
            for g, base in zip(o.param_groups, lrs): g["lr"] = base * scale
        for o in opts: o.zero_grad(set_to_none=True)

        spike_val = 0.0
        for _ in range(c["grad_accum"]):
            ids, lbl, msk = next(loader)
            # 2026-10-22: pinned + non_blocking (same values): the copy no longer stalls the CPU behind the GPU queue
            if device == "cuda":
                ids = ids.pin_memory().to(device, non_blocking=True); lbl = lbl.pin_memory().to(device, non_blocking=True)
            else:
                ids = ids.to(device); lbl = lbl.to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                # return_stats=True gives us the spike rate for logging without
                # a second forward pass. Extra cost is negligible.
                out = model(ids, return_stats=True)
                logits, sr = out[0], out[1]
                # Autoregressive shift: logits at t predict token at t+1.
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = lbl[:, 1:].contiguous()
                loss = F.cross_entropy(shift_logits.reshape(-1, c["vocab"]),
                                        shift_labels.reshape(-1), ignore_index=-100)
                # 2026-09-26: optional MQAR recall rehearsal (the pretraining aux,
                # train_gpu.mqar_aug_batch). v3 SFT without it lost most of the
                # base's recall. Off unless mqar_aug: true (all SFT configs = false).
                if c.get("mqar_aug", False):
                    from train_gpu import mqar_aug_batch
                    N_mq = int(torch.randint(c.get("mqar_N_min", 4), c.get("mqar_N_max", 32) + 1, (1,)).item())
                    x_mq, tgt_mq = mqar_aug_batch(c["batch_size"], N_mq, device)
                    loss = loss + c.get("mqar_weight", 0.5) * F.cross_entropy(model(x_mq)[:, -1, :], tgt_mq)
                if sr is not None:
                    # detach: sr may have requires_grad=True; explicit detach
                    # silences the PyTorch warning and doesn't affect training.
                    # 2026-10-22: kept on the GPU (float() here was a GPU sync every micro-step); read at log time
                    spike_val = spike_val + sr.detach().float() / c["grad_accum"]
            (loss / c["grad_accum"]).backward()
            if xc_frac > 0 and xc_rng.rand() < xc_frac:
                N_x = int(xc_rng.randint(c.get("mqar_N_min", 4), c.get("mqar_N_max", 32) + 1))
                x_x, tgt_x = mqar_gap_batch(xc_B, N_x, xc_C - 2 * N_x + int(xc_rng.randint(0, xc_C)), xc_data, xc_rng, device)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                    l_x = xc_w * F.cross_entropy(model(x_x)[:, -1, :].float(), tgt_x)
                (l_x / c["grad_accum"]).backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        for o in opts: o.step()
        step += 1; tokens += per_step_tok

        if step % c["log_every"] == 0:
            dt = time.time() - win; win = time.time()
            vram = (f" | vram {torch.cuda.memory_allocated()/1024**3:.1f}/"
                    f"{torch.cuda.memory_reserved()/1024**3:.1f} GB"
                    if device == "cuda" else "")
            spike_val = float(spike_val)
            spike_str = f" | spike {spike_val:.3f}" if spike_val > 0 else ""
            print(f"step {step} | {tokens/1e6:.1f}M tok | loss {loss.item():.3f}{spike_str} "
                  f"| {c['log_every']*per_step_tok/dt:,.0f} tok/s{vram}", flush=True)
        if step % c["ckpt_every"] == 0:
            save_ckpt(ckdir, model, opts, step, tokens, c)
        if gen_check_every > 0 and step % gen_check_every == 0:
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp and c.get("lora_base_bf16", False)):
                sample = gen_check(model, encode, gen_decode, gen_check_prompt,
                                    40, c["block_size"], device)
            print(f"  [gen-check] {sample!r}", flush=True)
        if preempt["hit"] and tokens < c["max_tokens"]:
            save_ckpt(ckdir, model, opts, step, tokens, c)
            print(f"preempted: saved step {step}, {tokens/1e6:.1f}M tok", flush=True)
            sys.exit(3)

    save_ckpt(ckdir, model, opts, step, tokens, c)
    print(f"done: {tokens/1e6:.1f}M tok in {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    main()
