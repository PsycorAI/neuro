"""Muon optimizer (Momentum Orthogonalized by Newton-Schulz).

Drop-in optimizer for 2D weight matrices (hidden-layer linear projections).
Empirically ~25-35% faster training to equal loss than AdamW at small scale,
state of the art on the NanoGPT speedrun benchmark in 2025-2026.

Use AdamW for embeddings, biases, LayerNorm scales -- anything that is not a
2D weight matrix between hidden layers.

Reference:
  Keller Jordan et al., "Muon: An optimizer for hidden layers in neural
  networks." https://kellerjordan.github.io/posts/muon/

This is a single-GPU implementation. The distributed variant is omitted.
"""
import torch


@torch.no_grad()
def _newton_schulz(G, steps=5, eps=1e-7):
    """Apply ~5 Newton-Schulz iterations to orthogonalize G in place.
    Coefficients tuned to converge fast in a few iterations (Keller Jordan).
    Operates in bfloat16 internally for speed; result cast back to G's dtype.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


@torch.no_grad()
def _newton_schulz_batched(G, steps=5, eps=1e-7):
    """_newton_schulz on a (k, m, n) stack of same-shape matrices: identical
    per-matrix math (per-matrix Frobenius norm), one bmm per product instead of
    k separate mm launches. For the tiny low-rank gate factors (16xN, dx16) the
    per-matrix path is pure launch overhead (~3.6ms each on a 5080)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    X = X / (X.norm(dim=(1, 2), keepdim=True) + eps)
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(1, 2)
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(1, 2)
    return X.to(G.dtype)


# batched_ns only stacks matrices up to this size: big matrices are compute-bound
# (nothing to gain) and stacking them would hold extra fp32 copies (VRAM).
BATCHED_NS_MAX_NUMEL = 1 << 20


class Muon(torch.optim.Optimizer):
    """Muon, optionally augmented with NorMuon's per-neuron adaptive LR.

    normuon=False (default): plain Muon. Byte-identical behavior to the
    pre-NorMuon implementation -- state dict has only "momentum_buffer",
    resumes cleanly from any pre-2026-07-09 checkpoint.

    normuon=True: NorMuon (arXiv:2510.05491). After Newton-Schulz produces
    the orthogonalized update U (same shape as p, orthonormal rows/cols),
    we track a per-row (per-output-neuron) 2nd-moment v_i via EMA of
    ||U_i||^2 / d_in, then divide U row-wise by sqrt(v_hat_i) + eps.
    Bias-corrected like Adam. Paper's reported win: +6% convergence at
    124M params, +15% at 350M, ~3% step-time overhead. Corrects Muon's
    tendency to leave certain neurons dominating.

    State dict gains "normuon_v" (per-row 2nd moment) and "normuon_step"
    (int) only when normuon=True; resumes with normuon=False stay clean.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 normuon=False, normuon_beta2=0.95, normuon_eps=1e-8,
                 weight_decay=0.0, batched_ns=False):
        if lr <= 0.0:
            raise ValueError(f"invalid lr: {lr}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        normuon=normuon, normuon_beta2=normuon_beta2, normuon_eps=normuon_eps,
                        weight_decay=weight_decay, batched_ns=batched_ns)
        super().__init__(params, defaults)
        # validate: every param must be a 2D matrix
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2:
                    raise ValueError(
                        f"Muon is for 2D weight matrices only; got shape {tuple(p.shape)}. "
                        f"Pass 1D params (embeddings, biases, LayerNorm) to AdamW instead.")

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            ns = group["ns_steps"]
            use_norm = group.get("normuon", False)
            b2 = group.get("normuon_beta2", 0.95)
            eps_n = group.get("normuon_eps", 1e-8)
            wd = group.get("weight_decay", 0.0)
            batched = group.get("batched_ns", False)
            buckets = {}   # shape -> [(p, update)] for the batched small-matrix path
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                # Nesterov lookahead
                update = g.add(buf, alpha=mom) if nesterov else buf
                if batched and p.numel() <= BATCHED_NS_MAX_NUMEL:
                    # clone: `update` may alias the momentum buffer (nesterov=False)
                    buckets.setdefault(tuple(p.shape), []).append((p, update.clone()))
                    continue
                # Orthogonalize via Newton-Schulz
                update = _newton_schulz(update, steps=ns)
                self._apply_update(p, update, lr, use_norm, b2, eps_n, wd)
            for items in buckets.values():
                U = _newton_schulz_batched(torch.stack([u for _, u in items]), steps=ns)
                for (p, _), update in zip(items, U.unbind(0)):
                    self._apply_update(p, update, lr, use_norm, b2, eps_n, wd)
        return loss

    @torch.no_grad()
    def _apply_update(self, p, update, lr, use_norm, b2, eps_n, wd):
        state = self.state[p]
        if use_norm:
            # NorMuon: per-row (per output-neuron) adaptive LR.
            # Scale-invariant statistic: mean-square across each row.
            row_ms = update.pow(2).mean(dim=1)   # shape (d_out,)
            if "normuon_v" not in state:
                state["normuon_v"] = torch.zeros_like(row_ms)
                state["normuon_step"] = 0
            v = state["normuon_v"]
            state["normuon_step"] += 1
            v.mul_(b2).add_(row_ms, alpha=1 - b2)
            # Adam-style bias correction (early steps under-scaled otherwise)
            v_hat = v / (1 - b2 ** state["normuon_step"])
            row_scale = (v_hat.sqrt() + eps_n).unsqueeze(1)   # (d_out, 1)
            # Per-neuron rebalancing inflates the global update norm by
            # ~sqrt(max_dim) (each row forced to unit RMS). NorMuon
            # (arXiv:2510.05491) renormalizes globally afterward; the
            # paper uses 0.2*O/RMS(O) to match Adam. Our baseline is Muon,
            # so we preserve the *orthogonalized* update's global RMS
            # instead -- keeps the global step identical to the Muon arm,
            # isolating NorMuon's neuron-balancing as the only A/B variable.
            orig_rms = update.square().mean().sqrt()
            update = update / row_scale
            new_rms = update.square().mean().sqrt()
            update = update * (orig_rms / (new_rms + eps_n))
        # LR scaled by sqrt(max(d_out, d_in) / min(d_out, d_in)) -- "muP-style"
        scale = max(1.0, (max(p.shape) / min(p.shape)) ** 0.5)
        # Decoupled weight decay -- Moonshot arXiv:2502.16982 technique #1,
        # the one thing this implementation was missing. Without it Muon's
        # weights grow without bound (measured 2.0-2.8x over 12k steps at
        # 276M), which pushes the GDN-2 gate logits past the sigmoid's
        # useful band and freezes the write/decay gates. Uses the UNSCALED
        # group lr so decay is uniform across matrix shapes; group["lr"] is
        # rewritten each step by the schedule, so decay tracks it.
        if wd != 0.0:
            p.mul_(1.0 - lr * wd)
        p.add_(update, alpha=-lr * scale)


def build_muon_adamw(model, muon_lr=0.02, adamw_lr=3e-4, weight_decay=0.0,
                    adamw_fused=True, adamw_8bit=False, fix_gate_matrices=False,
                    normuon=False, normuon_beta2=0.95, normuon_eps=1e-8,
                    route_embeddings_by_type=False, precise_gate_filter=False,
                    head_weight_decay=None, muon_mem_lr=None, batched_ns=False,
                    muon_attn_lr=None, no_decay_names=None):
    """Split model parameters and return (muon, adamw) optimizers.

    no_decay_names (2026-10-12, default None = byte-identical grouping): AdamW parameters whose name
    contains one of these substrings go to the no-decay group (e.g. ["embed.weight"]). Found on the 1B
    v4 run: the token embedding sits in the decay group, and decoupled decay over 227k steps shrank it
    13x by 25% of training (13L / 12k steps: 1.4x). The groups keep their order and count, only their
    membership changes -> resume from a checkpoint saved without it via remap_adamw_state().

    2D matrices in hidden layers -> Muon.
    Everything else (embeddings, biases, LayerNorm, output head if you want it
    in AdamW) -> AdamW.

    Convention: the output head (lm_head / classifier) and the input embedding
    are kept in AdamW because they interact with vocabulary size.

    fix_gate_matrices (default False, legacy): when True, also excludes
    degenerate (1,N) matrices (W_beta, W_alpha, titans' W_theta/W_eta/W_alpha_t,
    MoR/rdepth routers.*) from Muon -- NS on a (1,N) row collapses to
    direction-normalization, a fixed-norm update regardless of gradient
    magnitude, silently wrong for exactly the gates that control write-
    strength/forget in the delta rule. Default is False (matches the param
    grouping of every checkpoint saved before 2026-07-06) so resumed runs'
    optimizer state still loads; set True for fresh runs only -- changes
    which params live in which optimizer, so it is NOT resume-compatible
    with a checkpoint saved under the other setting.

    route_embeddings_by_type (default False, F2 code-review finding
    2026-07-12): the substring `"embed" in name.lower()` predicate misses any
    `nn.Embedding` module whose attribute name does not literally contain
    "embed" -- notably `EngramV2.bucket` (an `nn.Embedding` of size
    K*n_tables x rank, e.g. 32768x64 at 30M or 65536x64 at 90M/500M). Muon's
    Newton-Schulz orthogonalization + muP scale on such a sparse-gradient
    lookup table is an obvious mismatch; the module's own docstring says
    embeddings belong in AdamW. When True, additionally routes any parameter
    whose owning module is `nn.Embedding` to AdamW. Same NOT-resume-compat
    caveat as fix_gate_matrices -- optimizer state layout differs.
    """
    if precise_gate_filter and not fix_gate_matrices:
        raise ValueError(
            "precise_gate_filter requires fix_gate_matrices: without it the "
            "degenerate (1,N) W_gate/W_beta/W_alpha matrices would fall "
            "through to Muon, where Newton-Schulz on a single row collapses "
            "to a fixed-magnitude update (the F1 pathology).")
    if route_embeddings_by_type:
        # Build id-based lookup of owning-module type for every parameter.
        # `named_parameters(recurse=False)` on each module gives us only that
        # module's OWN params, so we can attribute each param to a single
        # owning-module type without ambiguity.
        param_to_owner = {}
        for _mname, mod in model.named_modules():
            for _pname, p in mod.named_parameters(recurse=False):
                param_to_owner[id(p)] = type(mod).__name__
    muon_params, adamw_params = [], []
    head_ids = set()
    mem_ids = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.dim() == 2 and (not fix_gate_matrices or min(p.shape) > 1)
        is_embedding = "embed" in name.lower() or "pos" in name.lower()
        if route_embeddings_by_type:
            is_embedding = is_embedding or (
                param_to_owner.get(id(p)) == "Embedding")
        # mtp_proj feeds directly into self.head (model.py: self.head(self.mtp_proj(...))),
        # so it is part of the output-head path. flagship_v2 diverged via runaway
        # mtp_proj-norm growth (+34%) dragging the head with it, because mtp_proj sat
        # in the general decay group at the too-weak global wd. Route it into the
        # dedicated head_weight_decay group too. (No effect when head_weight_decay is
        # None -> head_decay stays empty -> byte-identical grouping.)
        is_head = ("head" in name.lower() or "lm_head" in name.lower()
                   or "mtp_proj" in name.lower())
        if is_head:
            head_ids.add(id(p))
        # Memory-path matrices for the optional decoupled memory LR
        # (muon_mem_lr): the associative read/write projections (W_k/W_v/W_q)
        # and the GDN-2 erase/write gates (W_b/W_w). W_fdecay lives in AdamW
        # and the scalar W_beta/W_alpha are 1-row -> AdamW, so this set is
        # exactly the Muon-trained tensors of the associative memory op.
        if name.endswith((".W_k.weight", ".W_v.weight", ".W_q.weight",
                          ".W_b.weight", ".W_w.weight")):
            mem_ids.add(id(p))
        # Gating params (write gate W_gate is (1,N) -> Newton-Schulz degenerate;
        # decay_raw is 1D) belong in AdamW, not Muon.
        # The substring test below was written for `decay_raw` (1D) and
        # `W_gate` (1xN). It ALSO matches `W_fdecay.weight` -- a full
        # (d_mem, n_neurons) matrix -- so GDN-2's decay gate has always been
        # trained by AdamW at `lr` while its erase/write siblings (W_b, W_w)
        # get Muon at `muon_lr` x muP scale: ~15x different step magnitude and
        # a different update geometry, on the one gate that controls
        # forgetting. See docs/pre_flagship_audit_2026_08_08.md B1.
        #
        # With fix_gate_matrices on, every OTHER name this predicate catches is
        # already excluded (1D params aren't matrices; W_gate/local_gate are
        # (1,N) degenerate), so precise_gate_filter=True moves exactly one
        # tensor per block -- W_fdecay.weight -- into Muon and nothing else.
        # Default False = every checkpoint through P4. NOT resume-compatible
        # across the flip (param-group layout changes).
        if precise_gate_filter:
            is_gate = False
        else:
            is_gate = "gate" in name.lower() or "decay" in name.lower()
        if is_matrix and not is_embedding and not is_head and not is_gate:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    if not muon_params:
        muon = None
    elif muon_attn_lr is not None:
        # Separate LR for the hybrid attention block's matrices (qkv / o_proj / ffn -- names only the
        # HybridAttentionBlock has). Same proportional-schedule mechanism as muon_mem_lr. Extra param
        # group -> NOT resume-compatible; fresh runs only. Mutually exclusive with muon_mem_lr.
        if muon_mem_lr is not None:
            raise ValueError("muon_attn_lr and muon_mem_lr are mutually exclusive")
        attn_ids = {id(p) for n, p in model.named_parameters()
                    if any(s in n for s in (".qkv.", ".o_proj.", ".ffn."))}
        a_muon = [p for p in muon_params if id(p) in attn_ids]
        b_muon = [p for p in muon_params if id(p) not in attn_ids]
        groups = [g for g in ({"params": b_muon, "lr": muon_lr}, {"params": a_muon, "lr": muon_attn_lr}) if g["params"]]
        print(f"  muon_attn_lr={muon_attn_lr}: {len(a_muon)} attention matrices, {len(b_muon)} others at {muon_lr}")
        muon = Muon(groups, lr=muon_lr, weight_decay=weight_decay,
                    normuon=normuon, normuon_beta2=normuon_beta2, normuon_eps=normuon_eps,
                    batched_ns=batched_ns)
    elif muon_mem_lr is not None:
        # Decoupled memory-path LR. The muon_lr recall-vs-ppl tradeoff
        # (neuro_muon_lr_recall_tradeoff) is monotonic in OPPOSITE
        # directions: low LR helps recall, high LR helps val_ppl, because
        # one global knob couples the memory path to the backbone. Give the
        # memory matrices their own (typically lower) LR and leave the
        # backbone at muon_lr -> tests whether low-LR-on-memory buys recall
        # WITHOUT the ppl cost. The schedule scales each group's base lr
        # proportionally (train_gpu base_lrs snapshot), so the ratio holds
        # through cosine. Extra param group changes optimizer-state layout
        # -> NOT resume-compatible; fresh runs only.
        mem_muon = [p for p in muon_params if id(p) in mem_ids]
        base_muon = [p for p in muon_params if id(p) not in mem_ids]
        groups = []
        if base_muon:
            groups.append({"params": base_muon, "lr": muon_lr})
        if mem_muon:
            groups.append({"params": mem_muon, "lr": muon_mem_lr})
        print(f"  muon_mem_lr={muon_mem_lr}: {len(mem_muon)} memory-path "
              f"matrices (W_k/v/q, W_b/w) decoupled from {len(base_muon)} "
              f"backbone matrices; NOT resume-compat")
        muon = Muon(groups, lr=muon_lr, weight_decay=weight_decay,
                    normuon=normuon, normuon_beta2=normuon_beta2, normuon_eps=normuon_eps,
                    batched_ns=batched_ns)
    else:
        muon = Muon(muon_params, lr=muon_lr, weight_decay=weight_decay,
                    normuon=normuon, normuon_beta2=normuon_beta2, normuon_eps=normuon_eps,
                    batched_ns=batched_ns)
    # Weight decay must hit only 2D weight matrices, NOT 1D params. Decaying
    # RMSNorm/LayerNorm gains, biases, gdn2_a, or gate_bias_f (which carries
    # the deliberate -3.0 slow-decay init) toward zero is harmful — the
    # standard nanoGPT/Moonshot split. The Muon group is all-2D so its wd is
    # already correct; only the mixed AdamW group needs splitting. When wd=0
    # this collapses to a single group == the old behavior (byte-identical).
    if weight_decay != 0.0 or head_weight_decay is not None:
        # Optional dedicated higher-WD group for the LM head only (default off ->
        # byte-identical grouping). The flagship's divergence failure mode is
        # runaway head-norm growth (neuro_flagship_divergence_root_cause); the
        # head already sits in the AdamW decay group, but decoupled decay at the
        # global wd (~lr*wd/step) was too weak to hold it and z_loss 1e-3 was
        # null. head_weight_decay splits the head into its own group so it can be
        # decayed harder without over-decaying the rest. NOTE: adding a param
        # group changes optimizer-state LAYOUT, so a config with
        # head_weight_decay set is NOT resume-compatible with a checkpoint saved
        # without it -- fresh runs only (same caveat as fix_gate_matrices).
        head_decay = ([p for p in adamw_params if id(p) in head_ids and p.dim() >= 2]
                      if head_weight_decay is not None else [])
        head_id_set = {id(p) for p in head_decay}
        nd_ids = ({id(p) for n, p in model.named_parameters() if any(s in n for s in no_decay_names)}
                  if no_decay_names else set()) - head_id_set
        decay = [p for p in adamw_params if p.dim() >= 2 and id(p) not in head_id_set and id(p) not in nd_ids]
        nodecay = [p for p in adamw_params if p.dim() < 2 or (id(p) in nd_ids and p.dim() >= 2)]
        if nd_ids:
            print(f"  no_decay_names={list(no_decay_names)}: {sum(1 for p in nodecay if p.dim() >= 2)} "
                  f"matrices moved to the no-decay AdamW group")
        adamw_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        if head_decay:
            adamw_groups.append({"params": head_decay, "weight_decay": head_weight_decay})
    else:
        adamw_groups = adamw_params
    adamw_kwargs = dict(lr=adamw_lr, weight_decay=weight_decay)
    if adamw_8bit:
        # 8-bit AdamW via bitsandbytes. ~4x smaller optimizer state, small
        # accuracy cost (usually <1% ppl at this scale). Ignores 'fused' flag.
        from bitsandbytes.optim import AdamW8bit
        adamw = AdamW8bit(adamw_groups, **adamw_kwargs)
    else:
        if adamw_fused and torch.cuda.is_available():
            adamw_kwargs["fused"] = True
        adamw = torch.optim.AdamW(adamw_groups, **adamw_kwargs)
    return muon, adamw, len(muon_params), len(adamw_params)


def remap_adamw_state(old_opt, new_opt, state):
    """Optimizer state saved under old_opt's grouping -> a state dict new_opt can load (2026-10-12).
    Both optimizers must hold the SAME parameter tensors, grouped differently (no_decay_names): the
    per-parameter moments follow their tensor, the group hyperparameters are new_opt's own."""
    old_idx = {id(p): i for i, p in enumerate(p for g in old_opt.param_groups for p in g["params"])}
    new_params = [p for g in new_opt.param_groups for p in g["params"]]
    assert set(old_idx) == {id(p) for p in new_params}, "remap_adamw_state: the two optimizers hold different parameters"
    return {"state": {j: state["state"][old_idx[id(p)]] for j, p in enumerate(new_params)
                      if old_idx[id(p)] in state["state"]},
            "param_groups": new_opt.state_dict()["param_groups"]}
