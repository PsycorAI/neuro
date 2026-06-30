"""Phase 2.5 GPU trainer.

Trains neuro (spiking Hebbian + optional SET) or a matched transformer on
streaming real tokens, with checkpoint/resume, eval (incl. a memory-ablation
probe), throughput, and an energy report. Config-driven (YAML).

  python src/train_gpu.py --config configs/phase25_a.yaml            # GPU run
  python src/train_gpu.py --config configs/phase25_a.yaml --device cpu --smoke
"""
import os, sys, time, math, argparse, glob
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from model import SpikingHebbianLM
from baseline import TinyTransformer
from data_stream import StreamText
from sparse import SET
import energy
import yaml

BIN = "/home/glenn/projects/bdh/data/tokenized/train.bin"

# MQAR-augmented training (Zoology/BASED-style): mix synthetic associative-recall
# into the LM loss so the model explicitly trains the in-context retrieval skill.
# Disjoint key/value token ranges (within our 16384 student vocab, matching
# scripts/recall_eval.py defaults for distribution consistency).
MQAR_KEY_LO, MQAR_KEY_HI = 1000, 6000
MQAR_VAL_LO, MQAR_VAL_HI = 8000, 14000


def mqar_aug_batch(B, N, device):
    """B sequences [k1 v1 ... kN vN kq], target = value paired with kq. Returns (x, target)."""
    xs = []; tgs = []
    for _ in range(B):
        keys = torch.randperm(MQAR_KEY_HI - MQAR_KEY_LO)[:N] + MQAR_KEY_LO
        vals = torch.randperm(MQAR_VAL_HI - MQAR_VAL_LO)[:N] + MQAR_VAL_LO
        seq = torch.stack([keys, vals], 1).reshape(-1)
        qi = int(torch.randint(0, N, (1,)))
        xs.append(torch.cat([seq, keys[qi:qi + 1]]))
        tgs.append(int(vals[qi]))
    return torch.stack(xs).to(device), torch.tensor(tgs, device=device)


def _build_optimizers(model, c, device):
    """Build a list of optimizer(s) based on config.

    optimizer: "adamw" (default) -> single fused AdamW
    optimizer: "muon"            -> Muon for 2D hidden-layer matrices,
                                    fused AdamW for embeddings/heads/biases/norms

    Returns a list so the training loop can iterate uniformly across either
    setup. Single-optimizer ckpts from before this change still resume cleanly.
    """
    kind = c.get("optimizer", "adamw").lower()
    if kind == "muon":
        from muon import build_muon_adamw
        muon, adamw, n_mat, n_other = build_muon_adamw(
            model,
            muon_lr=c.get("muon_lr", 0.02),
            adamw_lr=c["lr"],
            adamw_fused=(device == "cuda"))
        print(f"optimizer: Muon ({n_mat} matrices, lr={c.get('muon_lr', 0.02)}) "
              f"+ AdamW ({n_other} other params, lr={c['lr']})")
        return [o for o in (muon, adamw) if o is not None]
    fused_kw = {"fused": True} if device == "cuda" else {}
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], **fused_kw)
    print(f"optimizer: AdamW{' (fused)' if fused_kw else ''} (lr={c['lr']})")
    return [opt]


def build_model(c):
    if c["arch"] == "spiking":
        return SpikingHebbianLM(c["vocab"], d=c["d"], n_neurons=c["n_neurons"],
                                d_mem=c["d_mem"], recurrent=c.get("recurrent", False),
                                rec_density=c.get("rec_density", 0.05),
                                compile_safe=c.get("compile", False),
                                tie_weights=c.get("tie_weights", False),
                                n_layers=c.get("n_layers", 1),
                                use_fpt=c.get("use_fpt", False),
                                fpt_K=c.get("fpt_K", 10),
                                beta=c.get("beta", 0.9),
                                lam=c.get("lam", 0.98),
                                eta=c.get("eta", 1.0),
                                learnable_decay=c.get("learnable_decay", False),
                                write_gate=c.get("write_gate", False),
                                delta_rule=c.get("delta_rule", False),
                                beta_floor=c.get("beta_floor", 0.0),
                                decay_gate=c.get("decay_gate", False),
                                titans=c.get("titans", False),
                                local_attn=c.get("local_attn", False),
                                local_window=c.get("local_window", 64),
                                n_heads=c.get("n_heads", 1),
                                pre_conv=c.get("pre_conv", False),
                                pre_conv_kernel=c.get("pre_conv_kernel", 4),
                                vector_beta=c.get("vector_beta", False),
                                mtp_enabled=c.get("mtp_enabled", False),
                                mtp_offset=c.get("mtp_offset", 2),
                                use_rmsnorm=c.get("use_rmsnorm", False),
                                embed_scale=c.get("embed_scale", False),
                                mtp_depth=c.get("mtp_depth", 1))
    return TinyTransformer(c["vocab"], d=c["d"], n_head=c["n_head"],
                           n_layer=c["n_layer"], max_T=c["block_size"])


def _save_ckpt(ckdir, raw_model, opts, step, tokens, c):
    p = f"{ckdir}/step_{step:07d}.pt"
    torch.save({"model": raw_model.state_dict(),
                "opt_states": [o.state_dict() for o in opts],
                "step": step, "tokens": tokens, "cfg": c}, p)
    for o in sorted(glob.glob(f"{ckdir}/step_*.pt"))[:-c.get("keep_last", 3)]:
        os.remove(o)
    print(f"  saved {p}")


def _set_lam(raw_model, lam):
    """Set the Hebbian decay on the model and all blocks (for lam warmup)."""
    raw_model.lam = lam
    for b in raw_model.blocks:
        b.lam = lam


@torch.no_grad()
def evaluate(model, data, c, device, iters=20):
    model.eval()
    amp = c["amp"] and device == "cuda"

    def ppl(ablate):
        tot = 0.0
        for _ in range(iters):
            x, y = data.batch(c["eval_batch"], c["block_size"], "val", device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                out = model(x, ablate_memory=ablate) if c["arch"] == "spiking" else model(x)
                tot += F.cross_entropy(out.reshape(-1, c["vocab"]), y.reshape(-1)).item()
        return math.exp(tot / iters)

    full = ppl(False)
    if c["arch"] == "spiking":
        x, _ = data.batch(c["eval_batch"], c["block_size"], "val", device)
        _, sr = model(x, return_stats=True)
        return full, ppl(True), sr.item()
    return full, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    with open(args.config) as f:
        c = yaml.safe_load(f)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch._dynamo.config.cache_size_limit = 64   # cap compile cache growth (~3-4 graphs needed)
    if args.smoke:
        c.update(block_size=16, batch_size=4, grad_accum=1, eval_batch=8,
                 max_tokens=20000, eval_every=200, ckpt_every=10**9, log_every=50, amp=False)
    c.setdefault("eval_batch", c["batch_size"])
    torch.manual_seed(c.get("seed", 0))

    bin_path = c.get("data_bin", BIN)
    data_orig_vocab = c.get("data_orig_vocab", 128256)
    data = StreamText(bin_path, vocab=c["vocab"], orig_vocab=data_orig_vocab)
    stateful = c.get("stateful_stream", False) and c["arch"] == "spiking"
    seq_stream = None
    if stateful:
        from data_stream import SequentialStream
        seq_stream = SequentialStream(bin_path, vocab=c["vocab"],
                                      orig_vocab=data_orig_vocab,
                                      batch_size=c["batch_size"],
                                      seed=c.get("seed", 0))
        print(f"STATEFUL streaming enabled: {c['batch_size']} contiguous cursors, "
              f"TBPTT across {c['block_size']}-token chunks")
    raw_model = build_model(c).to(device)
    nparams = sum(p.numel() for p in raw_model.parameters())
    opts = _build_optimizers(raw_model, c, device)

    # ---- Optional: knowledge distillation from a Llama teacher --------------
    kd_teacher = None
    kd_lut = None
    if c.get("kd_enabled", False) and c["arch"] == "spiking":
        from kd import load_teacher, load_lut
        kd_teacher = load_teacher(c.get("kd_teacher", "meta-llama/Llama-3.2-1B"),
                                   device=device)
        kd_lut = load_lut(c["vocab"]).to(device)
        print(f"KD enabled: teacher={c.get('kd_teacher','meta-llama/Llama-3.2-1B')} "
              f"alpha={c.get('kd_alpha',0.5)} T={c.get('kd_temperature',1.0)}")
    setms = None
    if c["arch"] == "spiking" and c.get("recurrent"):
        setms = [(b, SET(b.rec_mask, c.get("set_zeta", 0.3)))
                 for b in raw_model.blocks if hasattr(b, "rec_mask")]
    if c.get("compile") and device == "cuda":
        mode = c.get("compile_mode", "default")  # "default" | "reduce-overhead" | "max-autotune"
        model = torch.compile(raw_model, mode=mode, dynamic=c.get("compile_dynamic", False))
        print(f"torch.compile enabled (mode={mode}) — first 1-2 steps will be slow (graph capture)")
    else:
        model = raw_model

    ckdir = f"/home/glenn/projects/neuro/models/checkpoints/{c['run_name']}"
    os.makedirs(ckdir, exist_ok=True)
    step, tokens = 0, 0
    cks = sorted(glob.glob(f"{ckdir}/step_*.pt"))
    if cks:
        s = torch.load(cks[-1], map_location=device, weights_only=False)
        missing, unexpected = raw_model.load_state_dict(s["model"], strict=False)
        if unexpected:
            print(f"  [info] ignoring {len(unexpected)} unused keys from ckpt "
                  f"(e.g. {unexpected[0]})")
        if missing:
            print(f"  [warn] {len(missing)} keys missing in ckpt: {missing[:3]}")
        if "opt_states" in s and len(s["opt_states"]) == len(opts):
            for o, st in zip(opts, s["opt_states"]):
                o.load_state_dict(st)
        elif "opt" in s and len(opts) == 1:
            opts[0].load_state_dict(s["opt"])               # legacy single-optimizer ckpt
        else:
            print("  [warn] optimizer state in ckpt incompatible with current config; starting fresh")
        step, tokens = s["step"], s["tokens"]
        print(f"resumed {cks[-1]} @ step {step}, {tokens/1e6:.1f}M tok")

    per_step_tok = c["batch_size"] * c["grad_accum"] * c["block_size"]
    total_steps = max(1, c["max_tokens"] // per_step_tok)
    warmup_steps = int(c.get("lr_warmup_frac", 0.02) * total_steps)
    lr_min_frac = c.get("lr_min_frac", 0.1)
    grad_clip = c.get("grad_clip", 1.0)

    def lr_scale(s):
        """Linear warmup, then cosine decay to lr_min_frac of peak."""
        if s < warmup_steps:
            return s / max(1, warmup_steps)
        p = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]

    # Optional lam (Hebbian decay) warmup: ramp from lam_start to lam_end over the
    # first lam_ramp_frac of training, then hold. Lets the model learn an easy
    # short-memory base early, then build long-lived memory late.
    lam_start = c.get("lam", 0.98)
    lam_end = c.get("lam_end", lam_start)
    lam_ramp_steps = int(c.get("lam_ramp_frac", 0.8) * total_steps)
    lam_warmup = (lam_end != lam_start)

    def lam_at(s):
        if not lam_warmup or s >= lam_ramp_steps:
            return lam_end
        return lam_start + (lam_end - lam_start) * (s / max(1, lam_ramp_steps))

    if lam_warmup:
        print(f"lam warmup: {lam_start} -> {lam_end} over {lam_ramp_steps} steps")

    print(f"arch={c['arch']} params={nparams/1e6:.2f}M device={device} "
          f"target={c['max_tokens']/1e9:.3f}B tok | {per_step_tok} tok/step "
          f"| total_steps≈{total_steps} | warmup={warmup_steps} | grad_clip={grad_clip}")
    if stateful and kd_teacher is not None:
        raise ValueError("stateful_stream and kd_enabled can't be combined yet "
                         "(KD needs raw IDs from the random sampler). Test separately.")
    t0 = time.time(); win = t0
    carry_state = None   # carried (M, mem, prev_spk) for stateful TBPTT
    use_sparsity = c.get("sparsity_lambda", 0.0) > 0 and c["arch"] == "spiking"
    while tokens < c["max_tokens"]:
        model.train()
        # Apply LR schedule for this step
        scale = lr_scale(step)
        for o, lrs in zip(opts, base_lrs):
            for g, base in zip(o.param_groups, lrs):
                g["lr"] = base * scale
        # Apply lam (Hebbian decay) warmup for this step
        if lam_warmup:
            cur_lam = lam_at(step)
            _set_lam(raw_model, cur_lam)
            c["lam"] = cur_lam   # saved ckpts/eval rebuild at the operating lam

        for o in opts: o.zero_grad(set_to_none=True)
        for _ in range(c["grad_accum"]):
            if stateful:
                x, y, reset = seq_stream.next(c["block_size"], device)
                # ST Phase 6: stochastic state reset (document-boundary proxy) —
                # with prob reset_prob per block, drop carried state so long-lived
                # memory dims don't bleed across unrelated documents. Combined
                # with the cursor-wrap reset.
                rp = c.get("reset_prob", 0.0)
                if rp > 0:
                    reset = reset | (torch.rand(c["batch_size"], device=device) < rp)
                # Zero the carried state for any cursor that wrapped or was reset
                if carry_state is not None and bool(reset.any()):
                    for lay in carry_state["layers"]:
                        lay["M"][reset] = 0
                        lay["mem"][reset] = 0
                        lay["prev_spk"][reset] = 0
                with torch.autocast(device, dtype=torch.bfloat16, enabled=c["amp"] and device == "cuda"):
                    if use_sparsity:
                        out, sr, carry_state = model(x, return_stats=True,
                                                     initial_state=carry_state,
                                                     return_final_state=True)
                    else:
                        out, carry_state = model(x, initial_state=carry_state,
                                                 return_final_state=True)
                        sr = None
                    main_loss = F.cross_entropy(out.reshape(-1, c["vocab"]), y.reshape(-1))
                    if sr is not None:
                        target = c.get("sparsity_target", 0.05)
                        main_loss = main_loss + c["sparsity_lambda"] * (sr - target).clamp(min=0)
                    loss = main_loss / c["grad_accum"]
                loss.backward()
                continue
            need_raw = kd_teacher is not None
            if need_raw:
                x, y, raw = data.batch(c["batch_size"], c["block_size"], "train",
                                        device, return_raw=True)
            else:
                x, y = data.batch(c["batch_size"], c["block_size"], "train", device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=c["amp"] and device == "cuda"):
                # Forward (with optional return_stats for sparsity penalty)
                if use_sparsity:
                    out, sr = model(x, return_stats=True)
                else:
                    out = model(x)
                    sr = None
                # Hard-target CE
                ce = F.cross_entropy(out.reshape(-1, c["vocab"]), y.reshape(-1))
                # Knowledge distillation
                if kd_teacher is not None:
                    from kd import teacher_distribution, kd_loss
                    teacher_probs = teacher_distribution(
                        kd_teacher, raw, kd_lut,
                        student_vocab=c["vocab"],
                        temperature=c.get("kd_temperature", 1.0))
                    main_loss = kd_loss(out, teacher_probs,
                                         alpha=c.get("kd_alpha", 0.5),
                                         target_ids=y,
                                         temperature=c.get("kd_temperature", 1.0))
                else:
                    main_loss = ce
                # Sparsity penalty
                if sr is not None:
                    target = c.get("sparsity_target", 0.05)
                    main_loss = main_loss + c["sparsity_lambda"] * (sr - target).clamp(min=0)
                # Z-loss regularization (PaLM / DeepSeek-V3 style):
                # penalty on (log Σ exp(logits))² keeps the softmax denominator
                # bounded — stabilizes training, especially under bf16/fp8.
                # Typical weight 1e-4.
                z_w = c.get("z_loss_weight", 0.0)
                if z_w > 0:
                    log_z = torch.logsumexp(out, dim=-1)
                    z_loss = (log_z ** 2).mean()
                    main_loss = main_loss + z_w * z_loss
                # Multi-Token Prediction (MTP) auxiliary head — DeepSeek-V3 style.
                # Predicts token at position t+mtp_offset (default t+2) from the
                # same hidden state used for next-token. Free quality lift,
                # gradient signal cost ~10% extra fwd FLOPs (one extra linear).
                if c.get("mtp_enabled", False) and raw_model._last_mtp_logits is not None:
                    offset = c.get("mtp_offset", 2)
                    # main logits at position t predict y[t] (token at t+1).
                    # mtp logits at position t predict token at t+offset.
                    # We supervise positions where t+offset is in range.
                    mtp_logits_aligned = raw_model._last_mtp_logits[:, : -(offset - 1)]
                    y_mtp = y[:, (offset - 1):]
                    mtp_ce = F.cross_entropy(
                        mtp_logits_aligned.reshape(-1, c["vocab"]),
                        y_mtp.reshape(-1))
                    main_loss = main_loss + c.get("mtp_weight", 0.3) * mtp_ce
                # MQAR-augmented training: explicit in-context recall pressure
                if c.get("mqar_aug", False):
                    n_lo = c.get("mqar_N_min", 4); n_hi = c.get("mqar_N_max", 32)
                    N_mq = int(torch.randint(n_lo, n_hi + 1, (1,)).item())
                    x_mq, tgt_mq = mqar_aug_batch(c["batch_size"], N_mq, device)
                    logits_mq = model(x_mq)[:, -1, :]
                    mqar_ce = F.cross_entropy(logits_mq, tgt_mq)
                    main_loss = main_loss + c.get("mqar_weight", 0.5) * mqar_ce
                loss = main_loss / c["grad_accum"]
            loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        for o in opts: o.step()
        step += 1; tokens += per_step_tok
        if setms and step % c.get("set_every", 200) == 0:
            for blk, s in setms:
                s.step(blk.W_rec.weight)
        if step % c["log_every"] == 0:
            dt = time.time() - win; win = time.time()
            vram = f" | vram {torch.cuda.memory_allocated()/1024**3:.1f}/{torch.cuda.memory_reserved()/1024**3:.1f} GB" if device == "cuda" else ""
            print(f"step {step} | {tokens/1e6:.1f}M tok | loss {loss.item()*c['grad_accum']:.3f} "
                  f"| {c['log_every']*per_step_tok/dt:,.0f} tok/s{vram}")
        if step % c["eval_every"] == 0:
            full, abl, sr = evaluate(raw_model, data, c, device)
            msg = f"  [eval] val_ppl {full:.2f}"
            if abl is not None:
                msg += f" | ablated_ppl {abl:.2f} (memory Δ={abl-full:+.2f}) | spike {sr:.3f}"
            print(msg, flush=True)
            # Brain snapshot for spiking models (cheap, useful for after-the-fact analysis)
            if c["arch"] == "spiking" and c.get("snapshot_brain", True):
                try:
                    bdir = f"/home/glenn/projects/neuro/models/brains/{c['run_name']}"
                    os.makedirs(bdir, exist_ok=True)
                    x, _ = data.batch(1, c["block_size"], "val", device)
                    with torch.no_grad():
                        _, brain_state = raw_model(x, return_final_state=True)
                    bpath = f"{bdir}/step_{step:07d}.brain"
                    raw_model.save_brain(brain_state, bpath)
                except Exception as e:
                    print(f"  [warn] brain snapshot failed: {e}")
            if device == "cuda":
                torch.cuda.empty_cache()   # release fragmented blocks freed by eval
        if step % c["ckpt_every"] == 0:
            _save_ckpt(ckdir, raw_model, opts, step, tokens, c)

    # Always save a FINAL checkpoint so the last training state is evaluable
    # (prevents the "evaluated at step 2000 while siblings are at 3000" confound
    # when max_tokens isn't a multiple of ckpt_every).
    _save_ckpt(ckdir, raw_model, opts, step, tokens, c)
    print(f"done: {tokens/1e6:.1f}M tok in {(time.time()-t0)/3600:.2f}h")
    if c["arch"] == "spiking":
        _, _, sr = evaluate(raw_model, data, c, device)
        energy.compare(model, sr, baseline_d=c["d"], baseline_layers=c.get("n_layer", 2),
                       vocab=c["vocab"], seq_lens=[c["block_size"], 512, 4096])


if __name__ == "__main__":
    main()
