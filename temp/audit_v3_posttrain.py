"""Mechanical key-by-key diff of the three new post-training configs against the
live pretrain reference, bucketed per the audit-config procedure."""
import yaml

REF = "/home/glenn/projects/neuro/configs/flagship_v3_363m.yaml"
CANDS = ["sft_v3_363m", "orpo_v3_363m", "opd_v3_363m"]

ARCH = ["arch", "vocab", "d", "n_neurons", "d_mem", "n_layers", "block_size",
        "attention_layer_positions", "attention_heads", "attention_max_seq_len",
        "attention_qk_norm", "recurrent", "recurrent_depth", "recurrent_bptt_k",
        "recurrent_router_enabled", "use_fpt", "fpt_K", "use_fla_kernel",
        "fused_ce", "delta_rule", "decay_gate", "gdn2_enabled", "gdn2_chunked",
        "gdn2_gate_norm", "osdn_enabled", "engram_v2_enabled", "engram_v2_K",
        "engram_v2_heads", "engram_v2_orders", "engram_v2_rank", "mtp_enabled",
        "mtp_offset", "mtp_weight", "mtp_depth", "use_rmsnorm", "embed_scale"]
RECIPE = ["beta_floor", "normuon", "muon_fix_gate_matrices",
          "muon_route_embeddings_by_type", "muon_precise_gate_filter",
          "normuon_beta2", "normuon_eps", "optimizer", "z_loss_weight"]
EXPECTED_DIFF = ["run_name", "max_tokens", "lr", "muon_lr", "weight_decay",
                 "mqar_aug", "mqar_weight", "sparsity_lambda", "batch_size",
                 "grad_accum", "ckpt_every", "log_every", "eval_every",
                 "lr_warmup_frac", "compile", "compile_mode", "compile_dynamic",
                 "head_weight_decay", "data_bin", "keep_last", "gdn2_chunk_size",
                 "gdn2_ckpt_chunks", "watchdog_dstd_thresh",
                 "watchdog_valppl_ema_alpha", "watchdog_valppl_tol",
                 "watchdog_valppl_patience", "kd_enabled", "snapshot_brain",
                 "curriculum_enabled", "curriculum_frac",
                 "curriculum_block_tokens", "mqar_N_min", "mqar_N_max",
                 "sparsity_target", "eval_batch", "lam", "lr_min_frac", "amp",
                 "seed", "grad_clip"]

ref = yaml.safe_load(open(REF))
print(f"reference: {REF}\n")
for name in CANDS:
    c = yaml.safe_load(open(f"/home/glenn/projects/neuro/configs/{name}.yaml"))
    print(f"{'='*72}\n{name}\n{'='*72}")
    fails = []
    for k in ARCH:
        if k not in c:
            fails.append(f"[FAIL] {k}: MISSING (reference: {ref.get(k)!r})")
        elif c[k] != ref.get(k):
            fails.append(f"[FAIL] {k}: candidate={c[k]!r} (reference: {ref.get(k)!r})")
    print("ARCH-CRITICAL:", "all match" if not fails else "")
    for f in fails:
        print("  " + f)

    warns = []
    for k in RECIPE:
        if k not in c:
            warns.append(f"[WARN] {k}: MISSING (reference: {ref.get(k)!r})")
        elif c[k] != ref.get(k):
            warns.append(f"[note] {k}: candidate={c[k]!r} (reference: {ref.get(k)!r})")
    print("RECIPE DEFAULTS:", "all present and matching" if not warns else "")
    for w in warns:
        print("  " + w)

    known = set(ARCH) | set(RECIPE) | set(EXPECTED_DIFF)
    unknown = sorted(k for k in c if k not in known)
    print(f"STAGE-SPECIFIC KEYS ({len(unknown)}): {unknown}")

    # token budget: post-training stages start a FRESH counter (model-only load),
    # so max_tokens is an absolute budget, not resume+delta.
    bs, ga = c["batch_size"], c["grad_accum"]
    if name.startswith("opd"):
        per = bs * ga * c["opd_max_new"]
        note = f"response tokens only (batch {bs} x accum {ga} x {c['opd_max_new']} new)"
    elif name.startswith("orpo"):
        per = bs * ga * c["dpo_max_len"] * 2
        note = f"chosen+rejected (batch {bs} x accum {ga} x {c['dpo_max_len']} x2)"
    else:
        per = bs * ga * c["sft_max_len"]
        note = f"batch {bs} x accum {ga} x {c['sft_max_len']}"
    print(f"TOKEN BUDGET: max_tokens={c['max_tokens']/1e6:.0f}M / {per} tok/step "
          f"=> ~{c['max_tokens']//per} steps  [{note}]")
    print()
