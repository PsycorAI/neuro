"""Phase 2.5 GPU trainer.

Trains neuro (spiking Hebbian + optional SET) or a matched transformer on
streaming real tokens, with checkpoint/resume, eval (incl. a memory-ablation
probe), throughput, and an energy report. Config-driven (YAML).

  python src/train_gpu.py --config configs/phase25_a.yaml            # GPU run
  python src/train_gpu.py --config configs/phase25_a.yaml --device cpu --smoke
"""
import os, sys, time, math, argparse, glob, signal
import numpy as np
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

# Watchdog state (persists across evaluate() calls in one process). Trips
# after two CONSECUTIVE evals meet a tripwire, so one noisy eval never aborts.
# Exit code 2 = watchdog abort; chain scripts use `;` so the chain continues.
# Rules (from docs/plan_2026_08_05.md P3):
#   dstd > 3.0 on any block   x2 evals  → collapse-in-progress (per gotcha #6)
#   at_floor > 20% on any blk x2 evals  → write-gate saturating
#   spike rising past 0.20    x2 evals  → sparsity penalty overwhelmed
# Enable via `watchdog_enabled: true` (default true when set is absent).
_WATCHDOG = {"dstd_hits": 0, "atfloor_hits": 0, "spike_hits": 0, "prev_spike": None,
             "valppl_hits": 0, "best_valppl": None, "best_step": None,
             "valppl_ema": None, "prev_wnorm": None}

# NaN/Inf guard state (added 2026-08-07). Skips optimizer.step() when the
# clipped gradient norm is non-finite (would otherwise write NaN weights and
# corrupt training silently — dstd/at_floor watchdogs cannot detect this
# because `nan > threshold` is False). Aborts with exit code 3 after 3
# consecutive non-finite steps.
_NAN_GUARD = {"consecutive": 0}

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


def mqar_gap_batch(B, N, g, data, rng, device):
    """Haystack MQAR (2026-09-28; 'interference wall', arXiv 2609.16183): [k1 v1 .. kN vN] + g tokens of real TRAIN
    text + [kq] -> target vq. Gap windows come from the private numpy RNG `rng` (LM data stream untouched); gap tokens
    equal to one of the sequence's keys are replaced by random ids in 6000-7999 (outside the key/value ranges)."""
    x, tg = mqar_aug_batch(B, N, "cpu")                      # (B, 2N+1) = kv block + query
    if g > 0:
        starts = rng.randint(0, data.split - g - 1, size=B)
        gap = torch.from_numpy(data.lut[np.stack([np.asarray(data.data[s:s + g], dtype=np.int64) for s in starts])])
        keys = x[:, 0:2 * N:2]
        clash = (gap.unsqueeze(-1) == keys.unsqueeze(1)).any(-1)
        gap = torch.where(clash, torch.randint(6000, 8000, gap.shape), gap)
        x = torch.cat([x[:, :2 * N], gap, x[:, 2 * N:]], 1)
    return x.to(device), tg.to(device)


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
        use_normuon = c.get("normuon", False)
        muon, adamw, n_mat, n_other = build_muon_adamw(
            model,
            muon_lr=c.get("muon_lr", 0.02),
            adamw_lr=c["lr"],
            weight_decay=c.get("weight_decay", 0.0),
            adamw_fused=(device == "cuda"),
            adamw_8bit=c.get("adamw_8bit", False),
            fix_gate_matrices=c.get("muon_fix_gate_matrices", False),
            normuon=use_normuon,
            normuon_beta2=float(c.get("normuon_beta2", 0.95)),
            normuon_eps=float(c.get("normuon_eps", 1e-8)),
            route_embeddings_by_type=c.get("muon_route_embeddings_by_type", False),
            precise_gate_filter=c.get("muon_precise_gate_filter", False),
            head_weight_decay=c.get("head_weight_decay"),
            muon_mem_lr=c.get("muon_mem_lr"),
            muon_attn_lr=c.get("muon_attn_lr"),
            batched_ns=c.get("muon_batched_ns", False))
        opt_name = "NorMuon" if use_normuon else "Muon"
        print(f"optimizer: {opt_name} ({n_mat} matrices, lr={c.get('muon_lr', 0.02)}) "
              f"+ {'AdamW8bit' if c.get('adamw_8bit', False) else 'AdamW'} "
              f"({n_other} other params, lr={c['lr']})")
        if c.get("head_weight_decay") is not None:
            print(f"  head_weight_decay={c.get('head_weight_decay')} "
                  f"(dedicated LM-head decay group; NOT resume-compat with runs without it)")
        return [o for o in (muon, adamw) if o is not None]
    if c.get("adamw_8bit", False):
        from bitsandbytes.optim import AdamW8bit
        opt = AdamW8bit(model.parameters(), lr=c["lr"],
                        weight_decay=c.get("weight_decay", 0.0))
        print(f"optimizer: AdamW8bit (lr={c['lr']})")
    else:
        fused_kw = {"fused": True} if device == "cuda" else {}
        opt = torch.optim.AdamW(model.parameters(), lr=c["lr"],
                                weight_decay=c.get("weight_decay", 0.0), **fused_kw)
        print(f"optimizer: AdamW{' (fused)' if fused_kw else ''} (lr={c['lr']})")
    return [opt]


def spikeglu_k_at(c, step, total_steps):
    """r16 b3 SpikeGLU k-ramp (2026-10-02, docs/v4_plan_review_2026_10_01.md F2): the active-unit budget is
    the full hidden width (dense bilinear GLU) until attn_ffn_spikeglu_ramp_start of training, then shrinks
    geometrically to attn_ffn_spikeglu_k at attn_ffn_spikeglu_ramp_end (default 0.9) and stays there --
    learn dense, sparsify late (as the LIF late sparsity ramp, r11c). None = no ramp (fixed k)."""
    rs = c.get("attn_ffn_spikeglu_ramp_start")
    if rs is None or not c.get("attn_ffn_spikeglu_k"):
        return None
    k, hidden = int(c["attn_ffn_spikeglu_k"]), int(c["attn_ffn_hidden"])
    re_ = float(c.get("attn_ffn_spikeglu_ramp_end", 0.9))
    f = min(1.0, max(0.0, (step / max(1, total_steps) - float(rs)) / max(1e-9, re_ - float(rs))))
    return int(round(hidden * (k / hidden) ** f))


def mlp_rate_target_at(c, step, total_steps):
    """r19 homeostatic MLP firing-rate target (2026-10-03): 1.0 (no threshold pressure) until attn_ffn_ramp_start,
    then geometric to attn_ffn_rate_target at attn_ffn_ramp_end (default 0.9) -- the same late schedule as the
    DSN k-ramp. None = no homeostatic threshold configured."""
    tgt = c.get("attn_ffn_rate_target")
    if not c.get("attn_ffn_homeo") or tgt is None:
        return None
    rs, re_ = float(c.get("attn_ffn_ramp_start", 0.6)), float(c.get("attn_ffn_ramp_end", 0.9))
    f = min(1.0, max(0.0, (step / max(1, total_steps) - rs) / max(1e-9, re_ - rs)))
    return float(tgt) ** f


def build_model(c):
    if c["arch"] == "spiking":
        m = SpikingHebbianLM(c["vocab"], d=c["d"], n_neurons=c["n_neurons"],
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
                                mtp_depth=c.get("mtp_depth", 1),
                                use_swiglu=c.get("use_swiglu", False),
                                depth_scaled_init=c.get("depth_scaled_init", False),
                                markov_enabled=c.get("markov_enabled", False),
                                markov_rank=c.get("markov_rank", 64),
                                confidence_enabled=c.get("confidence_enabled", False),
                                compile_inv=c.get("compile_inv", False),
                                use_fla_kernel=c.get("use_fla_kernel", False),
                                gdn2_min_heads=c.get("gdn2_min_heads", 1),
                                gdn2_chunked=c.get("gdn2_chunked", True),
                                gdn2_ckpt_chunks=c.get("gdn2_ckpt_chunks", True),
                                gdn2_chunk_size=c.get("gdn2_chunk_size", 64),
                                gdn2_gate_norm=c.get("gdn2_gate_norm", False),
                                gate_rank=c.get("gate_rank", 0),
                                lif_exact=c.get("lif_exact", False),
                                lif_reset=float(c.get("lif_reset", 1.0)),
                                ff_conv=c.get("ff_conv", 0),
                                swa_window=c.get("swa_window", 0),
                                swa_heads=c.get("swa_heads", 4),
                                swa_dim=c.get("swa_dim", 64),
                                pre_norm=c.get("pre_norm", False),
                                mem_out_gate=c.get("mem_out_gate", 0),
                                lif_levels=c.get("lif_levels", 1),
                                lif_signed=c.get("lif_signed", False),
                                lif_learn_beta=c.get("lif_learn_beta", False),
                                lif_alpha=c.get("lif_alpha", None),
                                spk_mlp=c.get("spk_mlp", 0),
                                canon_a=c.get("canon_a", 0),
                                act_ckpt=c.get("act_ckpt", False),
                                gdn2_loglinear=c.get("gdn2_loglinear", False),
                                gdn2_loglinear_L=c.get("gdn2_loglinear_L", None),
                                n_experts=c.get("n_experts", 1),
                                route_topk=c.get("route_topk", None),
                                route_noise_std=c.get("route_noise_std", 0.0),
                                router_z_weight=c.get("router_z_weight", 0.0),
                                ponder_enabled=c.get("ponder_enabled", False),
                                ponder_steps=c.get("ponder_steps", 3),
                                ponder_capacity=c.get("ponder_capacity", None),
                                ponder_threshold=c.get("ponder_threshold", 0.5),
                                engram_enabled=c.get("engram_enabled", False),
                                engram_n=c.get("engram_n", 3),
                                engram_K=c.get("engram_K", 32768),
                                engram_rank=c.get("engram_rank", 64),
                                engram_v2_enabled=c.get("engram_v2_enabled", False),
                                engram_v2_orders=c.get("engram_v2_orders", (2, 3)),
                                engram_v2_K=c.get("engram_v2_K", 65536),
                                engram_v2_heads=c.get("engram_v2_heads", 2),
                                engram_v2_rank=c.get("engram_v2_rank", 64),
                                recurrent_depth=c.get("recurrent_depth", 1),
                                recurrent_bptt_k=c.get("recurrent_bptt_k", 0),
                                recurrent_router_enabled=c.get("recurrent_router_enabled", False),
                                recurrent_router_capacity=c.get("recurrent_router_capacity", None),
                                recurrent_router_threshold=c.get("recurrent_router_threshold", 0.5),
                                recurrent_deep_supervision=c.get("recurrent_deep_supervision", False),
                                recurrent_pass_weights=c.get("recurrent_pass_weights", None),
                                fracconn_enabled=c.get("fracconn_enabled", False),
                                fracconn_n=c.get("fracconn_n", 2),
                                gdn2_enabled=c.get("gdn2_enabled", False),
                                osdn_enabled=c.get("osdn_enabled", False),
                                osdn_eta=c.get("osdn_eta", 0.003),
                                osdn_dmin=c.get("osdn_dmin", 0.5),
                                osdn_dmax=c.get("osdn_dmax", 2.0),
                                sdm_enabled=c.get("sdm_enabled", False),
                                sdm_N=c.get("sdm_N", None),
                                sdm_W=c.get("sdm_W", 4),
                                sdm_R=c.get("sdm_R", 4),
                                attention_layer_positions=c.get("attention_layer_positions", None),
                                attention_heads=c.get("attention_heads", 8),
                                attention_max_seq_len=c.get("attention_max_seq_len", 4096),
                                attention_qk_norm=c.get("attention_qk_norm", True),
                                spike_rate_undiluted=c.get("spike_rate_undiluted", False),
                                final_norm=c.get("final_norm", False),
                                deltaproduct_nh=c.get("deltaproduct_nh", 1),
                                attn_single_residual=c.get("attn_single_residual", False),
                                telemetry_eval_only=c.get("telemetry_eval_only", False),
                                spk_layerscale=float(c.get("spk_layerscale", 0.0)),
                                attn_ffn=c.get("attn_ffn", True),
                                attn_ffn_spikeglu_k=c.get("attn_ffn_spikeglu_k", 0),
                                attn_ffn_hidden=c.get("attn_ffn_hidden", None),
                                attn_topk=c.get("attn_topk", 0),
                                engram_v2_hash=c.get("engram_v2_hash", "lo"),
                                spk_out_norm=c.get("spk_out_norm", True),
                                attn_out_gate=c.get("attn_out_gate", False),
                                attn_canon=c.get("attn_canon", 0),
                                attn_ffn_spikeglu_lif=c.get("attn_ffn_spikeglu_lif", False),
                                attn_ffn_act=c.get("attn_ffn_act", "gelu"),
                                attn_ffn_homeo=c.get("attn_ffn_homeo"))
        # attn_window_fixed (2026-10-04, default 0 = byte-identical): every attention block uses a fixed causal
        # sliding window of W keys at train AND eval (set here so every eval script that builds the model inherits
        # it). Purpose: give the memory layer a job attention cannot do (G3: with full 512 windows in 11 attention
        # layers the hippocampal layer went silent). Not combined with the stochastic window (attn_window_short).
        _wf = int(c.get("attn_window_fixed", 0) or 0)
        if _wf:
            assert not c.get("attn_window_short"), "attn_window_fixed and attn_window_short are exclusive"
            for _b in m.blocks:
                if hasattr(_b, "qkv"):
                    _b.attn_window = _wf
        # attn_chunk (r25, 2026-10-04, default 0 = byte-identical): attention only inside aligned C-token chunks, at
        # train AND eval (set here so every eval script inherits it); the memory spans the whole block_size sequence
        # with full gradient. See HybridAttentionBlock.attn_chunk.
        _ac = int(c.get("attn_chunk", 0) or 0)
        if _ac:
            assert not (c.get("attn_window_short") or _wf), "attn_chunk is exclusive with attn_window_*"
            for _b in m.blocks:
                if hasattr(_b, "qkv"):
                    _b.attn_chunk = _ac
        return m
    return TinyTransformer(c["vocab"], d=c["d"], n_head=c["n_head"],
                           n_layer=c["n_layer"], max_T=c["block_size"])


_ckpt_thread = None   # tracks the in-flight async save so we can join it
_pinned_cache = {}    # state_dict-path -> reusable pinned CPU staging tensor


def _to_cpu(obj, path=()):
    """Recursively clone tensors to CPU (dicts/lists/tuples walked, everything
    else returned as-is). Needed for BOTH model and optimizer state_dicts --
    optimizer state (exp_avg, exp_avg_sq, ...) holds live GPU tensors that
    opt.step() mutates in-place next step, which would race with a background
    save thread if left un-cloned. CUDA tensors copy into a pinned CPU
    destination (page-locked memory transfers over PCIe substantially faster
    than pageable) -- confirmed ~3.9x on a synthetic multi-GB benchmark.
    Pinned buffers are cached and reused by state_dict-key path across saves
    instead of freshly allocated every call: torch's pin_memory allocator
    never returns freed pinned blocks to the OS, so fresh-allocating every
    checkpoint grows Windows' WDDM "shared GPU memory" pool without bound
    (visible as VRAM spillover after the first eval+save). Safe to mutate
    the cache in place because _save_ckpt joins any in-flight save thread
    before calling this. CPU tensors (e.g. --device cpu smoke runs) fall
    back to .cpu().clone(), which is a genuine copy in that case too
    (torch.Tensor.cpu() is a no-op passthrough only when the source is
    already CPU, which would otherwise alias the live tensor)."""
    if torch.is_tensor(obj):
        if obj.is_cuda:
            pinned = _pinned_cache.get(path)
            if pinned is None or pinned.shape != obj.shape or pinned.dtype != obj.dtype:
                try:
                    pinned = torch.empty_like(obj, device="cpu", pin_memory=True)
                except RuntimeError:
                    # 2026-09-26: pinned host alloc raised "CUDA error: out of memory"
                    # on the 1B probe's final save and killed the run. A plain
                    # (pageable) copy is slower but always saves the checkpoint.
                    return obj.detach().cpu().clone()
                _pinned_cache[path] = pinned
            pinned.copy_(obj)
            return pinned
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v, path + (k,)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v, path + (i,)) for i, v in enumerate(obj))
    return obj


def _valppl_watchdog_step(state, full, step, c):
    """Pure val_ppl divergence decision (unit-tested in temp/test_valppl_watchdog.py).

    Mutates `state` (best_valppl/best_step/valppl_hits/valppl_ema) in place and
    returns (log_line_or_None, tripped_bool). Trips when the comparison signal
    exceeds its running best by > watchdog_valppl_tol for watchdog_valppl_patience
    consecutive evals, past the grace window. Catches the flagship divergence mode
    (runaway head norm -> rising val_ppl) that the dstd/at_floor/spike wires were
    all blind to. Side effects (print/save/exit) stay at the call site. See memory
    neuro_flagship_divergence_root_cause.

    watchdog_valppl_ema_alpha (default 0.0 = off, raw val_ppl) smooths the signal:
    signal = alpha*val_ppl + (1-alpha)*prev. Set >0 for runs whose val_ppl is
    noisy around a plateau, where a single lucky-low eval otherwise anchors `best`
    and makes normal ±few-% eval noise false-trip a healthy run (flagship_v3,
    2026-09-16). Smoothing keeps the real monotonic runaway (it survives the EMA)
    while spikes that revert within an eval or two do not.
    """
    alpha = c.get("watchdog_valppl_ema_alpha", 0.0)
    if alpha and alpha > 0.0:
        prev = state.get("valppl_ema")
        signal = full if prev is None else alpha * full + (1.0 - alpha) * prev
        state["valppl_ema"] = signal
    else:
        signal = full
    best = state["best_valppl"]
    if best is None or signal < best:
        state["best_valppl"], state["best_step"] = signal, step
        state["valppl_hits"] = 0
        return None, False
    tol = c.get("watchdog_valppl_tol", 0.02)
    over = signal > best * (1.0 + tol)
    state["valppl_hits"] = state["valppl_hits"] + 1 if over else 0
    line = (f"  [valppl] best {best:.2f} @ step {state['best_step']} | "
            f"now {signal:.2f}" + (f" (ema; raw {full:.2f})" if alpha else "")
            + f" ({(signal / best - 1) * 100:+.1f}%)"
            + (f" | DIVERGING x{state['valppl_hits']}" if over else ""))
    tripped = bool(c.get("watchdog_enabled", True)
                   and step > c.get("watchdog_grace_steps", 1500)
                   and state["valppl_hits"] >= c.get("watchdog_valppl_patience", 2))
    return line, tripped


def _save_ckpt(ckdir, raw_model, opts, step, tokens, c, async_save=True):
    """Save a checkpoint. GPU->CPU copy happens synchronously (unavoidable --
    the live tensors keep changing next step), but the disk write (the slow
    part, ~seconds for a multi-GB file) runs in a background thread so the
    GPU doesn't sit idle waiting on I/O."""
    global _ckpt_thread
    _join_ckpt_thread()   # must finish before _to_cpu reuses pinned buffers;
                           # also bounds thread count to 1 and guarantees the
                           # final synchronous save can't race a prior async write
    p = f"{ckdir}/step_{step:07d}.pt"
    blob = {"model": _to_cpu(raw_model.state_dict(), ("model",)),
            "opt_states": [_to_cpu(o.state_dict(), ("opt", i)) for i, o in enumerate(opts)],
            "step": step, "tokens": tokens, "cfg": c}

    def _write():
        tmp = p + ".tmp"
        torch.save(blob, tmp)
        os.replace(tmp, p)   # atomic on same filesystem -- a kill mid-write
                             # leaves only a .tmp, never a truncated step_*.pt
        for o in sorted(glob.glob(f"{ckdir}/step_*.pt"))[:-c.get("keep_last", 3)]:
            os.remove(o)
        print(f"  saved {p}")

    if async_save:
        import threading
        _ckpt_thread = threading.Thread(target=_write, daemon=False)
        _ckpt_thread.start()
    else:
        _write()


def _join_ckpt_thread():
    global _ckpt_thread
    if _ckpt_thread is not None:
        _ckpt_thread.join()
        _ckpt_thread = None


def _set_lam(raw_model, lam):
    """Set the Hebbian decay on the model and all blocks (for lam warmup)."""
    raw_model.lam = lam
    for b in raw_model.blocks:
        b.lam = lam


# Fixed eval RNG seed (unrelated to training's own RNG / c["seed"]): every
# evaluate() call across every config/run/step draws the SAME sequence of
# validation windows. Two effects: (1) within a run, successive evals measure
# progress on a fixed yardstick instead of a different random sample each
# time; (2) across runs/configs (A/B comparisons), the "which windows got
# sampled" confound is eliminated -- a huge source of the 10-15 ppl noise
# that swamped MoE/Engram/recurrent-depth screens. Bumped default iters
# 20->100 (5x eval compute, ~2-4% of total run time) for a further sqrt(5)
# reduction in sampling variance. Legacy numbers recorded before this fix
# (e.g. the 81.20 control) are NOT directly comparable -- re-run controls
# alongside any new screen that needs a fair baseline.
EVAL_SEED = 20260706


@torch.no_grad()
def evaluate(model, data, c, device, iters=None):
    model.eval()
    amp = c["amp"] and device == "cuda"
    iters = iters if iters is not None else c.get("eval_iters", 100)
    rng_state = np.random.get_state()

    def ppl(ablate):
        np.random.seed(EVAL_SEED)   # same windows for full and ablated passes
        tot = 0.0
        for _ in range(iters):
            x, y = data.batch(c["eval_batch"], c["block_size"], "val", device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                out = model(x, ablate_memory=ablate) if c["arch"] == "spiking" else model(x)
                tot += F.cross_entropy(out.reshape(-1, c["vocab"]), y.reshape(-1)).item()
        return math.exp(tot / iters)

    try:
        full = ppl(False)
        if c["arch"] == "spiking":
            x, _ = data.batch(c["eval_batch"], c["block_size"], "val", device)
            _, sr = model(x, return_stats=True)
            return full, ppl(True), sr.item()
        return full, None, None
    finally:
        np.random.set_state(rng_state)   # don't perturb training's own RNG stream


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
        if c.get("compile", False):
            # Preserve autocast-boundary bf16<->fp32 casts across fused kernels.
            # Without this, inductor drops the down/up-conversions the GDN-2
            # kernel's internal `torch.autocast(enabled=False)` context relies
            # on (bf16 on the triangular solve NaN'd training in earlier F7
            # investigation). This is the fix ezyang's Aug-2025 "state of
            # torch.compile for training" post explicitly names for this
            # exact autocast-boundary pattern.
            import torch._inductor.config as _ind
            _ind.emulate_precision_casts = True
    if args.smoke:
        c.update(block_size=16, batch_size=4, grad_accum=1, eval_batch=8,
                 max_tokens=20000, eval_every=200, ckpt_every=10**9, log_every=50, amp=False)
    c.setdefault("eval_batch", c["batch_size"])
    torch.manual_seed(c.get("seed", 0))
    np.random.seed(c.get("seed", 0))   # StreamText.batch draws train-window order from this

    bin_path = c.get("data_bin", BIN)
    data_orig_vocab = c.get("data_orig_vocab", 128256)
    data = StreamText(bin_path, vocab=c["vocab"], orig_vocab=data_orig_vocab,
                      independent_windows=c.get("data_independent_windows", False),
                      val_tokens=c.get("data_val_tokens"), prelut=c.get("data_prelut", False))
    if c.get("curriculum_enabled", False):
        from curriculum import CurriculumStream
        per_step_tok = c["batch_size"] * c["grad_accum"] * c["block_size"]
        total_steps = max(1, c["max_tokens"] // per_step_tok)
        warmup_calls = int(total_steps * c.get("curriculum_frac", 0.1)) * c["grad_accum"]
        data = CurriculumStream(data, bin_path, warmup_calls=warmup_calls,
                                 block_tokens=c.get("curriculum_block_tokens", 4096))
        print(f"curriculum warmup enabled: {warmup_calls} train batches drawn "
              f"easy->hard ({c.get('curriculum_frac', 0.1):.0%} of {total_steps} steps)")
    stateful = c.get("stateful_stream", False) and c["arch"] == "spiking"
    # Recurrent depth only engages when forward() gets initial_state=None;
    # stateful streaming passes a carried state every step, which would
    # SILENTLY disable the recurrent loop (and its router). Fail loudly instead.
    if c.get("recurrent_depth", 1) > 1 and stateful:
        raise ValueError("recurrent_depth > 1 is silently bypassed under "
                         "stateful_stream (the initial_state path is single-pass); "
                         "unset one of them")
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
                                   device=device,
                                   compile=c.get("kd_compile", False),
                                   compile_mode=c.get("kd_compile_mode", "reduce-overhead"),
                                   int8=c.get("kd_teacher_int8", False))
        kd_lut = load_lut(c["vocab"]).to(device)
        print(f"KD enabled: teacher={c.get('kd_teacher','meta-llama/Llama-3.2-1B')} "
              f"alpha={c.get('kd_alpha',0.5)} T={c.get('kd_temperature',1.0)}")
    setms = None
    if c["arch"] == "spiking" and c.get("recurrent"):
        setms = [(b, SET(b.rec_mask, c.get("set_zeta", 0.3)))
                 for b in raw_model.blocks if hasattr(b, "rec_mask")]
    if c.get("compile") and device == "cuda":
        mode = c.get("compile_mode", "default")  # "default" | "reduce-overhead" | "max-autotune"
        dynamic = c.get("compile_dynamic", False)
        if c.get("compile_regional", False):
            # TorchTitan-style regional compilation: compile each Hebbian block
            # in isolation instead of the whole LM. The LM wrapper stays eager,
            # so all its side-effect attribute writes (`_last_mtp_logits`,
            # `_last_moe_aux_loss`, etc.) execute normally in Python. Only the
            # compute-heavy block forwards get compiled, and since all blocks
            # share the same architecture, dynamo caches and reuses one graph.
            # 2026-10-03 FIX: the LM calls block.forward_sequence(), never block.forward(), so wrapping the block
            # module in torch.compile compiled NOTHING (blocks ran eager; g4ga8rc_c0s1 = 189 tok/s with the MQAR
            # whole-model compile stacked on top, 35.7 GB reserved -> WDDM spill). Compile the bound
            # forward_sequence of each block instead (as temp/proto_graphs.py compiles its per-block adapters).
            # Blocks share code, so raise dynamo's per-code recompile limit above the block count.
            import torch._dynamo as _dynamo      # (a bare `import torch._dynamo` would make `torch` local to main)
            _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 4 * len(raw_model.blocks))
            for b in raw_model.blocks:
                b.forward_sequence = torch.compile(b.forward_sequence, mode=mode, dynamic=dynamic)
            model = raw_model
            print(f"torch.compile REGIONAL (per-block, mode={mode}) — first 1-2 steps slow")
        else:
            model = torch.compile(raw_model, mode=mode, dynamic=dynamic)
            print(f"torch.compile enabled (mode={mode}) — first 1-2 steps will be slow (graph capture)")
    else:
        model = raw_model

    # MQAR uses a SEPARATE compiled wrapper (always mode="default", never
    # reduce-overhead/cudagraphs) rather than the main `model`. Two reasons:
    # (1) MQAR's sequence length varies with N in [mqar_N_min, mqar_N_max]
    #     every call -- dynamic=True lets one wrapper handle all shapes
    #     without retriggering full recompiles or (worse) cudagraph replay
    #     mismatches; (2) reduce-overhead's CUDA Graph capture is confirmed
    #     broken under repeated same-step invocations (grad_accum + this
    #     second forward triggered pytorch#169545 in practice), so this path
    #     stays off cudagraphs unconditionally regardless of compile_mode.
    # Previously MQAR called raw_model (fully eager) -- that cost ~59% of a
    # main-forward's time every step with zero speedup; compiling it here
    # (still real dev work, not free) is expected to recover most of that.
    mqar_model = raw_model
    _mqar_gap_rng = np.random.RandomState(int(c.get("seed", 0)) + 7919)   # haystack MQAR only (mqar_gap_max)
    # r25 nlfact (2026-10-05, default off): natural-language fact episodes, facts in attention chunk A and the question
    # in chunk B (src/nl_fact_aux.py; own RNG, so every other stream is untouched). Needs attn_chunk.
    _nlf = None
    if int(c.get("nlfact_every", 0) or 0):
        from nl_fact_aux import NLFactAux
        assert int(c.get("attn_chunk", 0) or 0), "nlfact_every needs attn_chunk"
        _nlf = NLFactAux(c["vocab"], c["attn_chunk"], seed=int(c.get("seed", 0)), k_max=int(c.get("nlfact_k_max", 4)),
                         in_frac=float(c.get("nlfact_in_frac", 0.3)),
                         subword_fallback=bool(c.get("lut_subword_fallback", False)))
    # Stochastic attention window (2026-09-29, SWAX arXiv 2509.24552; docs/contiguous_training_design.md): each step
    # the attention layers use a short sliding window w.p. attn_window_p, else full causal; the last
    # attn_window_long_frac of steps and every eval use full. The draw is keyed on (seed, step) -- no RNG stream is
    # consumed, so data/MQAR draws stay identical to the control and a resume reproduces the same windows.
    _win_short = int(c.get("attn_window_short", 0))
    _attn_blocks = [b for b in raw_model.blocks if hasattr(b, "qkv")] if _win_short else []
    if _win_short:
        print(f"stochastic attention window: {_win_short} w.p. {c.get('attn_window_p', 0.5)}, else full; "
              f"full for the last {c.get('attn_window_long_frac', 0.1):.0%} of steps and at eval "
              f"({len(_attn_blocks)} attention blocks)")
    # r16 b3: SpikeGLU blocks whose active-unit budget follows spikeglu_k_at each step (empty = no ramp).
    _sglu_blocks = ([b for b in raw_model.blocks if hasattr(getattr(b, "ffn", None), "k_cur")]
                    if spikeglu_k_at(c, 0, 1) is not None else [])
    if _sglu_blocks:
        print(f"SpikeGLU k-ramp: dense ({c['attn_ffn_hidden']}) until {c['attn_ffn_spikeglu_ramp_start']:.0%}, "
              f"then geometric to k={c['attn_ffn_spikeglu_k']} by {c.get('attn_ffn_spikeglu_ramp_end', 0.9):.0%} "
              f"({len(_sglu_blocks)} blocks)")
    # r19: sparse MLP activations (SparseAct); _homeo_acts = those with a homeostatic threshold (empty = off).
    from hybrid_attn import SparseAct
    _sparse_acts = [m for m in raw_model.modules() if isinstance(m, SparseAct)]
    _homeo_acts = [m for m in _sparse_acts if m.homeo] if mlp_rate_target_at(c, 0, 1) is not None else []
    if _sparse_acts:
        print(f"sparse MLP: {c.get('attn_ffn_act')} x{len(_sparse_acts)}"
              + (f", homeostatic {c['attn_ffn_homeo']} threshold: rate 1.0 until {c.get('attn_ffn_ramp_start', 0.6):.0%}, "
                 f"then geometric to {c['attn_ffn_rate_target']} by {c.get('attn_ffn_ramp_end', 0.9):.0%} "
                 f"(lr {c.get('attn_ffn_homeo_lr', 0.01)})" if _homeo_acts else ""))
    if c.get("gdn2_cumsum_eager_max_b"):
        import delta_chunked as _dc                     # Inductor SplitScan workaround (see delta_chunked.py)
        _dc.CUMSUM_EAGER_MAX_B = int(c["gdn2_cumsum_eager_max_b"])
    if c.get("compile") and device == "cuda" and c.get("mqar_aug", False) and c.get("compile_regional", False):
        # 2026-10-03: regional mode -- raw_model's blocks are already compiled; a second whole-model compile on top
        # doubles the graphs/memory. MQAR shapes stay fixed with mqar_pad_to (else each MQAR length recompiles).
        print("MQAR path uses the regionally compiled blocks (set mqar_pad_to to avoid per-length recompiles)")
    elif c.get("compile") and device == "cuda" and c.get("mqar_aug", False):
        # mqar_compile_static (2026-10-04, default off = byte-identical): dynamic=False for the MQAR wrapper. With
        # mqar_pad_to (+ mqar_gap_pad_to) the MQAR shapes are fixed anyway (65 / 512), and the dynamic=True graph hit
        # an Inductor codegen crash on the 512-token gap batches (r21c, InductorError get_block_shape NoneType).
        mqar_model = torch.compile(raw_model, mode="default", dynamic=not c.get("mqar_compile_static", False))
        print("torch.compile enabled for MQAR path (mode=default, dynamic=True)")

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
        if not c.get("lora_enabled", False) and "opt_states" in s and len(s["opt_states"]) == len(opts):
            try:
                for o, st in zip(opts, s["opt_states"]):
                    o.load_state_dict(st)
            except (ValueError, KeyError) as e:
                # Cross-stage resume (e.g. DPO Muon ckpt into pretrain NorMuon
                # trainer): outer count matches but internal param-group shapes
                # differ, so load_state_dict raises. Skip and start optimizer
                # fresh rather than crash.
                print(f"  [warn] optimizer state shape mismatch (cross-stage resume?); starting fresh: {e}")
        elif not c.get("lora_enabled", False) and "opt" in s and len(opts) == 1:
            try:
                opts[0].load_state_dict(s["opt"])           # legacy single-optimizer ckpt
            except (ValueError, KeyError) as e:
                print(f"  [warn] legacy opt state shape mismatch; starting fresh: {e}")
        else:
            if c.get("lora_enabled", False):
                print("  [info] LoRA enabled; starting optimizer fresh (base ckpt optimizer state skipped)")
            else:
                print("  [warn] optimizer state in ckpt incompatible with current config; starting fresh")
        step, tokens = s["step"], s["tokens"]
        print(f"resumed {cks[-1]} @ step {step}, {tokens/1e6:.1f}M tok")
        # Advance the data-side RNG and curriculum counter past what was already
        # trained on. Without this, a crash-resume replays random windows from
        # step 0 (halving effective data diversity for the remainder) and, if
        # curriculum is on, restarts the easy-block draw from the top at
        # mid-run LR. seed+step gives a deterministic new stream keyed to
        # where we resumed. CurriculumStream's per-call counter is advanced to
        # its already-consumed count (grad_accum train-window draws per step).
        np.random.seed(c.get("seed", 0) + step)
        if hasattr(data, "_train_calls"):
            data._train_calls = step * c["grad_accum"]
        # Stateful streams: move the cursors past what was already trained on (2026-09-29; before this a resume
        # replayed each stream from its start). The carried memory state is not checkpointed -> one reset.
        if seq_stream is not None:
            seq_stream.skip(step * c["grad_accum"], c["block_size"])

    per_step_tok = c["batch_size"] * c["grad_accum"] * c["block_size"]
    total_steps = max(1, c["max_tokens"] // per_step_tok)
    warmup_steps = int(c.get("lr_warmup_frac", 0.02) * total_steps)
    lr_min_frac = c.get("lr_min_frac", 0.1)
    grad_clip = c.get("grad_clip", 1.0)
    # lr_schedule: "cosine" (default, preserves every existing config byte-for-
    # byte) or "wsd" (warmup -> stable at 1.0 -> linear decay to lr_min_frac
    # over the last wsd_decay_frac of total_steps). WSD lets the endpoint be
    # chosen late: extend/stop by adjusting max_tokens without re-tuning the
    # shape. Quality ties cosine at a known fixed budget.
    lr_schedule = c.get("lr_schedule", "cosine")
    wsd_decay_frac = c.get("wsd_decay_frac", 0.15)
    decay_start = total_steps - int(wsd_decay_frac * (total_steps - warmup_steps))

    def lr_scale(s):
        """Warmup + (cosine|wsd) per lr_schedule."""
        if s < warmup_steps:
            return s / max(1, warmup_steps)
        if lr_schedule == "cosine":
            p = (s - warmup_steps) / max(1, total_steps - warmup_steps)
            return lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        if lr_schedule == "wsd":
            if s < decay_start:
                return 1.0                                            # stable
            p = (s - decay_start) / max(1, total_steps - decay_start)  # linear decay
            return lr_min_frac + (1 - lr_min_frac) * (1 - min(1.0, p))
        raise ValueError(f"lr_schedule must be 'cosine' or 'wsd', got {lr_schedule!r}")

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

    sched_desc = (f"wsd(decay_start={decay_start})" if lr_schedule == "wsd"
                  else "cosine")
    print(f"arch={c['arch']} params={nparams/1e6:.2f}M device={device} "
          f"target={c['max_tokens']/1e9:.3f}B tok | {per_step_tok} tok/step "
          f"| total_steps≈{total_steps} | warmup={warmup_steps} "
          f"| lr_sched={sched_desc} | grad_clip={grad_clip}")
    if stateful and kd_teacher is not None:
        raise ValueError("stateful_stream and kd_enabled can't be combined yet "
                         "(KD needs raw IDs from the random sampler). Test separately.")
    t0 = time.time(); win = t0
    carry_state = None   # carried (M, mem, prev_spk) for stateful TBPTT
    use_sparsity = c.get("sparsity_lambda", 0.0) > 0 and c["arch"] == "spiking"
    # Wall-clock-bounded chain runs (e.g. `timeout --signal=INT`) rely on SIGINT
    # reaching a clean save. Python's default handler raises KeyboardInterrupt,
    # which can land mid-backward() and skip the "Always save a FINAL checkpoint"
    # code below entirely (observed: w4_biggerbatch lost 6600-4000=2600 steps
    # to this on 2026-08-05). This handler just sets a flag instead of raising,
    # so the current step finishes normally and the loop exits into the
    # existing final-save/energy-report path below.
    _interrupted = {"hit": False}
    def _sigint_handler(signum, frame):
        _interrupted["hit"] = True
        print("\n[SIGINT] received — finishing current step, then saving final ckpt + exiting", flush=True)
    signal.signal(signal.SIGINT, _sigint_handler)
    while tokens < c["max_tokens"] and not _interrupted["hit"]:
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
        if _win_short:
            _long = step >= (1.0 - float(c.get("attn_window_long_frac", 0.1))) * total_steps
            _draw = np.random.RandomState((int(c.get("seed", 0)) * 1000003 + step) % (2 ** 32)).rand()
            _w = _win_short if (not _long and _draw < float(c.get("attn_window_p", 0.5))) else 0
            for _b in _attn_blocks: _b.attn_window = _w
        if _sglu_blocks:
            _kc = spikeglu_k_at(c, step, total_steps)
            for _b in _sglu_blocks: _b.ffn.k_cur = _kc
        if _homeo_acts:
            _rt = mlp_rate_target_at(c, step, total_steps)
            for _m in _homeo_acts: _m.rate_target = _rt
        for _micro in range(c["grad_accum"]):
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
                # (M rows are B * n_state_units, so repeat the per-row mask to match).
                if carry_state is not None and bool(reset.any()):
                    for lay in carry_state["layers"]:
                        lay["M"][reset.repeat_interleave(lay["M"].shape[0] // reset.numel())] = 0
                        lay["mem"][reset] = 0
                        lay["prev_spk"][reset] = 0
            need_raw = kd_teacher is not None
            if stateful:
                pass                                   # x, y already drawn from the contiguous streams above
            elif need_raw:
                x, y, raw = data.batch(c["batch_size"], c["block_size"], "train",
                                        device, return_raw=True)
            else:
                x, y = data.batch(c["batch_size"], c["block_size"], "train", device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=c["amp"] and device == "cuda"):
                # Forward (with optional return_stats for sparsity penalty).
                # return_mtp=True passes the MTP head's logits back as an extra
                # tuple element so downstream MTP loss composition doesn't have
                # to reach into raw_model._last_mtp_logits — that side-effect
                # attribute write is unsafe under whole-model torch.compile
                # (observed as MTP loss silently dropping under reduce-overhead,
                # ~30% training-signal loss, spike rate collapsing 0.185 -> 0.11).
                want_mtp = c.get("mtp_enabled", False)
                if stateful:
                    # Carried-state (TBPTT) forward; the returned state is detached (model.py). 2026-09-29: this
                    # path now falls through to the SHARED loss below (fused CE, MQAR aux, z-loss, MTP) -- the
                    # June branch computed plain CE only and skipped MQAR recall training.
                    _ret = model(x, return_stats=use_sparsity, initial_state=carry_state,
                                 return_final_state=True, return_mtp=want_mtp)
                    out = _ret[0]
                    sr = _ret[1] if use_sparsity else None
                    carry_state = _ret[2 if use_sparsity else 1]
                    mtp_logits = _ret[-1] if want_mtp else None
                elif use_sparsity:
                    if want_mtp:
                        out, sr, mtp_logits = model(x, return_stats=True, return_mtp=True)
                    else:
                        out, sr = model(x, return_stats=True)
                        mtp_logits = None
                else:
                    if want_mtp:
                        out, mtp_logits = model(x, return_mtp=True)
                    else:
                        out = model(x)
                        mtp_logits = None
                    sr = None
                # Knowledge distillation. NOTE: kd_loss computes the hard-target
                # CE internally (via target_ids), so we must NOT also compute a
                # standalone CE here when KD is on — that was a dead full-vocab
                # softmax every microstep. Only compute CE on the non-KD path.
                if kd_teacher is not None:
                    from kd import teacher_distribution, kd_loss
                    teacher_topk = teacher_distribution(
                        kd_teacher, raw, kd_lut,
                        student_vocab=c["vocab"],
                        temperature=c.get("kd_temperature", 1.0))
                    main_loss = kd_loss(out, teacher_topk,
                                         alpha=c.get("kd_alpha", 0.5),
                                         target_ids=y,
                                         temperature=c.get("kd_temperature", 1.0))
                else:
                    # Hard-target CE. Optional fused-CE (Liger) avoids materializing
                    # the full (B, T, V) logits gradient — saves VRAM, bit-exact.
                    if c.get("fused_ce", False):
                        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
                        if not hasattr(main, "_liger_ce"):
                            main._liger_ce = LigerCrossEntropyLoss()
                        main_loss = main._liger_ce(out.reshape(-1, c["vocab"]), y.reshape(-1))
                    else:
                        main_loss = F.cross_entropy(out.reshape(-1, c["vocab"]), y.reshape(-1))
                # Sparsity penalty
                if sr is not None:
                    target = c.get("sparsity_target", 0.05)
                    lam_sp = c["sparsity_lambda"]
                    # Late sparsity ramp (2026-09-30, default off = byte-identical): lambda stays at sparsity_lambda
                    # until sparsity_ramp_start_frac of training, then rises linearly to sparsity_lambda_final at the
                    # end -- learn first, sparsify late (gradual-sparsification practice).
                    _rs = c.get("sparsity_ramp_start_frac")
                    if _rs is not None:
                        _f = min(1.0, max(0.0, (step / total_steps - float(_rs)) / max(1e-9, 1.0 - float(_rs))))
                        lam_sp = lam_sp + _f * (float(c.get("sparsity_lambda_final", lam_sp)) - lam_sp)
                    main_loss = main_loss + lam_sp * (sr - target).clamp(min=0)
                # Z-loss regularization (PaLM / DeepSeek-V3 style):
                # penalty on (log Σ exp(logits))² keeps the softmax denominator
                # bounded — stabilizes training, especially under bf16/fp8.
                # Typical weight 1e-4.
                z_w = c.get("z_loss_weight", 0.0)
                if z_w > 0:
                    log_z = torch.logsumexp(out, dim=-1)
                    z_loss = (log_z ** 2).mean()
                    main_loss = main_loss + z_w * z_loss
                    # The MTP head is a SECOND output head and was entirely
                    # outside this guard. On the 363M flagship mtp_proj.0 crept
                    # +9% while head.weight was still flat, then blew up +28%
                    # alongside it -- the unguarded head may be the leading
                    # indicator. Default OFF to keep prior screens comparable;
                    # enable for the next flagship after an A/B.
                    if c.get("z_loss_include_mtp", False) and mtp_logits is not None:
                        main_loss = main_loss + z_w * (
                            torch.logsumexp(mtp_logits, dim=-1) ** 2).mean()
                # Batch C: Confidence head — BCE loss vs correctness (argmax==y).
                # Self-supervised signal for later inference-time gating.
                if c.get("confidence_enabled", False) and raw_model._last_confidence is not None:
                    with torch.no_grad():
                        correct = (out.argmax(dim=-1) == y).float()
                    conf_loss = F.binary_cross_entropy_with_logits(
                        raw_model._last_confidence, correct)
                    main_loss = main_loss + c.get("confidence_weight", 0.05) * conf_loss
                # Hebbian-MoE load-balance loss (Switch-style importance loss) --
                # keeps router mass from collapsing onto one expert.
                if c.get("n_experts", 1) > 1 and raw_model._last_moe_aux_loss is not None:
                    main_loss = main_loss + c.get("moe_aux_weight", 0.01) * raw_model._last_moe_aux_loss
                # Adaptive-computation (MoR expert-choice) router aux loss --
                # calibrates each router's score to predict its own top-k
                # membership so the causal per-token threshold used at eval matches
                # the training-time top-k. Compute budget itself is fixed by the
                # capacity schedule (hard top-k), NOT by this loss -- so unlike the
                # old ponder_cost_weight it has no collapse knife-edge.
                if c.get("ponder_enabled", False) and raw_model._last_ponder_aux is not None:
                    main_loss = main_loss + c.get("ponder_aux_weight", 0.01) * raw_model._last_ponder_aux
                # Recurrent-depth Stage 2 router aux loss -- same MoR calibration
                # role as the ponder aux above, but for RecurrentDepthRouter
                # (gates passes of the actual Hebbian block stack, not a bolt-on).
                if c.get("recurrent_router_enabled", False) and raw_model._last_rdepth_aux is not None:
                    main_loss = main_loss + c.get("recurrent_router_aux_weight", 0.1) * raw_model._last_rdepth_aux
                # Phase 1 deep supervision: CE loss at each NON-final recurrent
                # pass, so early passes get direct training signal as decodable
                # outputs instead of only ever being refined by a downstream
                # pass's gradient. Weights default to uniform 0.3 per pass if
                # not given (one entry per non-final pass, pass order).
                if c.get("recurrent_deep_supervision", False) and raw_model._last_pass_logits:
                    pass_weights = c.get("recurrent_pass_weights", None) or \
                        [0.3] * len(raw_model._last_pass_logits)
                    for pass_logits, w in zip(raw_model._last_pass_logits, pass_weights):
                        main_loss = main_loss + w * F.cross_entropy(
                            pass_logits.reshape(-1, c["vocab"]), y.reshape(-1))
                # Multi-Token Prediction (MTP) auxiliary head — DeepSeek-V3 style.
                # Predicts token at position t+mtp_offset (default t+2) from the
                # same hidden state used for next-token. Free quality lift,
                # gradient signal cost ~10% extra fwd FLOPs (one extra linear).
                if c.get("mtp_enabled", False) and mtp_logits is not None:
                    offset = c.get("mtp_offset", 2)
                    # main logits at position t predict y[t] (token at t+1).
                    # mtp logits at position t predict token at t+offset.
                    # We supervise positions where t+offset is in range.
                    mtp_logits_aligned = mtp_logits[:, : -(offset - 1)]
                    y_mtp = y[:, (offset - 1):]
                    if c.get("fused_ce", False):
                        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
                        if not hasattr(main, "_liger_ce"):
                            main._liger_ce = LigerCrossEntropyLoss()
                        mtp_ce = main._liger_ce(
                            mtp_logits_aligned.reshape(-1, c["vocab"]),
                            y_mtp.reshape(-1))
                    else:
                        mtp_ce = F.cross_entropy(
                            mtp_logits_aligned.reshape(-1, c["vocab"]),
                            y_mtp.reshape(-1))
                    main_loss = main_loss + c.get("mtp_weight", 0.3) * mtp_ce
                # MQAR-augmented training: explicit in-context recall pressure.
                # mqar_cadence (default 1) fires MQAR every N-th step at N*
                # mqar_weight, preserving the expected gradient (identical mean,
                # ~sqrt(cadence) higher variance) while amortizing the second
                # forward pass over multiple steps. Cadence>1 recipe-preserving
                # requires 30M screen confirmation; default 1 = pre-2026-07-16
                # behavior (every step).
                mqar_cad = int(c.get("mqar_cadence", 1))
                # mqar_once_per_step (2026-10-03, default off = byte-identical): the MQAR aux runs on the FIRST
                # micro-step only, with batch x grad_accum and weight x grad_accum -> same MQAR samples per optimizer
                # step and the same expected gradient, but 1 small launch-bound pass instead of grad_accum of them
                # (1B GA8: the per-micro aux cost ~32% of the step; mqar off 16.7k vs 11.3k tok/s).
                _mq_mult = c["grad_accum"] if c.get("mqar_once_per_step", False) else 1
                if c.get("mqar_aug", False) and (step % mqar_cad == 0) and (_mq_mult == 1 or _micro == 0):
                    n_lo = c.get("mqar_N_min", 4); n_hi = c.get("mqar_N_max", 32)
                    # mqar_N_power biases sampling toward high N (the recall
                    # failure region). power=1.0 -> uniform (byte-identical to
                    # torch.randint). power>1 -> u**(1/power) skews u toward 1,
                    # so N skews toward n_hi. Uniform gives trivial N=4 the same
                    # budget as N=28-32 where recall collapses; this reweights
                    # training toward the hard end. High-N batches are longer,
                    # so throughput drops as power rises.
                    n_pow = float(c.get("mqar_N_power", 1.0))
                    if n_pow == 1.0:
                        N_mq = int(torch.randint(n_lo, n_hi + 1, (1,)).item())
                    else:
                        u = float(torch.rand(1).item()) ** (1.0 / n_pow)
                        N_mq = int(round(n_lo + (n_hi - n_lo) * u))
                    # Static-shape MQAR (config-gated): snap N to a fixed bucket
                    # set so the 2N+1 seq length takes few distinct values ->
                    # stable compiled graphs (CUDAGraphs-compatible). Unset =
                    # byte-identical dynamic behavior. docs/training_speed_plan_2026_09.md 4.5
                    _buckets = c.get("mqar_buckets")
                    if _buckets:
                        N_mq = min(_buckets, key=lambda b: abs(b - N_mq))
                    _gmax = int(c.get("mqar_gap_max", 0))
                    # r16 b6 late-only gap dose (2026-10-02; default off = byte-identical): before
                    # mqar_gap_start_frac of training the aux uses plain MQAR at mqar_weight; from then on the
                    # haystack (gap) MQAR at mqar_gap_weight (default mqar_weight), its gap ramp measured from
                    # the start step. Gap batches cost ~2x step time, so a late dose cuts that cost.
                    _gs = c.get("mqar_gap_start_frac")
                    _g0 = 0 if _gs is None else float(_gs) * total_steps
                    _mq_w = c.get("mqar_weight", 0.5)
                    _xc = False
                    if _gmax > 0 and step >= _g0:
                        _rf = float(c.get("mqar_gap_ramp_frac", 0.0))
                        _cur = _gmax if _rf <= 0 else max(1, int(_gmax * min(1.0, (step - _g0) / max(1.0, _rf * total_steps))))
                        # r25 mqar_xchunk_frac (2026-10-04, default 0 = byte-identical, RNG stream untouched): with this
                        # probability the gap batch puts the kv block in attention chunk A and the query in chunk B
                        # (gap = attn_chunk - 2N + offset in [0, attn_chunk)), so only the memory can answer: trains
                        # beyond-the-window recall. Batch mqar_xchunk_batch (default mqar_batch), padded to 2*attn_chunk.
                        _xf = float(c.get("mqar_xchunk_frac", 0) or 0)
                        # mqar_xchunk_start_frac (2026-10-05, default 0 = unchanged): no xchunk batches before this
                        # fraction of training (let in-window recall emerge first).
                        _xc = (_xf > 0 and step >= float(c.get("mqar_xchunk_start_frac", 0.0)) * total_steps
                               and _mqar_gap_rng.rand() < _xf)
                        if _xc:
                            _C = int(c["attn_chunk"])
                            g_mq = _C - 2 * N_mq + int(_mqar_gap_rng.randint(0, _C))
                            _B_mq = int(c.get("mqar_xchunk_batch", c.get("mqar_batch", c["batch_size"])))
                        else:
                            g_mq = int(_mqar_gap_rng.randint(0, _cur + 1))
                            _B_mq = int(c.get("mqar_batch", c["batch_size"]))
                        x_mq, tgt_mq = mqar_gap_batch(_B_mq * _mq_mult, N_mq, g_mq,
                                                      data, _mqar_gap_rng, device)
                        if _gs is not None:
                            _mq_w = c.get("mqar_gap_weight", _mq_w)
                    else:
                        # mqar_aug_batch (r25, 2026-10-05, default batch_size = byte-identical): keeps the plain MQAR
                        # batch when T doubles and batch_size halves (p25a: recall never emerged at batch 16).
                        x_mq, tgt_mq = mqar_aug_batch(int(c.get("mqar_aug_batch", c["batch_size"])) * _mq_mult,
                                                      N_mq, device)
                    # mqar_model: separate compiled wrapper (mode=default,
                    # dynamic=True, never cudagraphs) -- see setup comment
                    # above. Falls back to raw_model (eager) when compile is
                    # off. Weights are shared with raw_model either way.
                    # mqar_pad_to (2026-10-02 speed plan; unset = byte-identical): right-pad to one fixed length so
                    # compile sees a single MQAR shape (each new length recompiled: GDN-2's chunk loop unrolls on T).
                    # Every layer is causal, so the pads cannot reach the query position read below.
                    _L_mq = x_mq.shape[1]
                    _pad_to = int(c.get("mqar_pad_to") or 0)
                    # mqar_gap_pad_to (2026-10-04, default unset = byte-identical): pad GAP batches (length up to
                    # 2N+1+gap) to their own fixed length and plain batches to mqar_pad_to -> two compiled shapes, and
                    # the 70% of training before the gap dose no longer pays for 512-token MQAR passes (r21b: 31.8 GB
                    # reserved -> WDDM spill, 7.8k vs 34k tok/s).
                    if c.get("mqar_gap_pad_to") and _L_mq > _pad_to:
                        _pad_to = int(c["mqar_gap_pad_to"])
                    if _xc:
                        _pad_to = 2 * int(c["attn_chunk"])
                    if _pad_to:
                        assert _L_mq <= _pad_to, f"mqar_pad_to {_pad_to} < MQAR length {_L_mq}"
                        x_mq = F.pad(x_mq, (0, _pad_to - _L_mq))
                    logits_mq = mqar_model(x_mq)[:, _L_mq - 1, :]
                    mqar_ce = F.cross_entropy(logits_mq, tgt_mq)
                    main_loss = main_loss + _mq_w * mqar_cad * _mq_mult * mqar_ce
                # r25 nlfact aux (default off): every nlfact_every-th step from nlfact_start_frac, first micro-step only.
                # Same compiled wrapper and padded length (2 * attn_chunk) as the xchunk MQAR batches.
                if (_nlf is not None and _micro == 0 and step % int(c["nlfact_every"]) == 0
                        and step >= float(c.get("nlfact_start_frac", 0.0)) * total_steps):
                    x_nf, last_nf, tgt_nf = _nlf.batch(int(c.get("nlfact_batch", 4)), data, device)
                    lg_nf = mqar_model(x_nf)[torch.arange(x_nf.shape[0], device=device), last_nf]
                    main_loss = main_loss + float(c.get("nlfact_weight", 0.15)) * int(c["nlfact_every"]) * \
                        F.cross_entropy(lg_nf, tgt_nf)
                loss = main_loss / c["grad_accum"]
            loss.backward()
        # NaN/Inf guard (added 2026-08-07). Without this, a numerical blowup
        # (autocast boundary, unstable delta-rule state, corrupt data batch)
        # produces NaN grads → clip_grad_norm_ returns NaN → opts.step()
        # writes NaN weights → all future forward passes NaN. The existing
        # dstd/at_floor/spike watchdogs cannot detect this (nan > threshold
        # is False). Skip step on non-finite grads, abort after 3 consecutive.
        if grad_clip > 0:
            gnorm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        else:
            gnorm = torch.tensor(0.0)
        if not torch.isfinite(gnorm):
            _NAN_GUARD["consecutive"] += 1
            n = _NAN_GUARD["consecutive"]
            print(f"  [NAN GUARD] non-finite grad norm at step {step} "
                  f"(loss={loss.item():.3g}); skipping optimizer step "
                  f"(consecutive hits: {n})", flush=True)
            if n >= 3:
                print(f"  [NAN GUARD] 3 consecutive non-finite grads — "
                      f"training diverged numerically. Saving pre-corruption "
                      f"ckpt (last good state, opts not stepped) and exiting 3.",
                      flush=True)
                _save_ckpt(ckdir, raw_model, opts, step, tokens, c, async_save=False)
                sys.exit(3)
            for o in opts: o.zero_grad(set_to_none=True)
        else:
            _NAN_GUARD["consecutive"] = 0
            for o in opts: o.step()
            for _m in _homeo_acts: _m.homeo_step(float(c.get("attn_ffn_homeo_lr", 0.01)))
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
            for _b in _attn_blocks: _b.attn_window = 0          # eval always uses the full window
            full, abl, sr = evaluate(raw_model, data, c, device)
            msg = f"  [eval] val_ppl {full:.2f}"
            if abl is not None:
                msg += f" | ablated_ppl {abl:.2f} (memory Δ={abl-full:+.2f}) | spike {sr:.3f}"
            print(msg, flush=True)
            # r16 DSN telemetry: current k-WTA budget and, for DSN-LIF, each layer's firing rate and mean leak.
            _dsn = [b.ffn for b in raw_model.blocks if hasattr(getattr(b, "ffn", None), "k_cur")]
            if _dsn and (_sglu_blocks or _dsn[0].lif):
                print("  [dsn] " + " ".join(
                    (f"rate={f._last_rate.item():.3f} beta={torch.sigmoid(f.beta_logit).mean().item():.3f}"
                     if f.lif and f._last_rate is not None else f"k={f.k if f.k_cur is None else f.k_cur}")
                    for f in _dsn), flush=True)
            # r19 sparse-MLP telemetry: each block's firing rate (last forward = eval) and mean threshold.
            if _sparse_acts:
                print("  [mlp] " + " ".join(
                    f"rate={m._last_rate.item():.3f}" + (f" thr={m.theta.mean().item():.3f}" if m.homeo else "")
                    for m in _sparse_acts if m._last_rate is not None)
                      + (f" | target {_homeo_acts[0].rate_target:.3f}" if _homeo_acts else ""), flush=True)
            # Output-head weight-norm telemetry (2026-08-25). The 363M flagship
            # diverged via runaway LM-head norm growth while every block and the
            # embedding decayed normally, and NOTHING in the log showed it -- it
            # was found only by post-hoc ckpt diffing. Cheap: a few param norms.
            # See memory neuro_flagship_divergence_root_cause.
            wn = {n: float(p.detach().float().pow(2).sum().sqrt())
                  for n, p in raw_model.named_parameters()
                  if p.dim() >= 2 and (n in ("head.weight", "embed.weight")
                                       or n.startswith("mtp_proj."))}
            if wn:
                prev_wn = _WATCHDOG["prev_wnorm"] or {}
                def _fmt_wn(n):
                    d = ((wn[n] - prev_wn[n]) / prev_wn[n] * 100) if prev_wn.get(n) else None
                    return (f"{n.replace('.weight', '')}={wn[n]:.2f}"
                            + (f" ({d:+.2f}%)" if d is not None else ""))
                print("  [wnorm] " + " | ".join(_fmt_wn(n) for n in sorted(wn)), flush=True)
                _WATCHDOG["prev_wnorm"] = wn
            # val_ppl divergence tripwire (2026-08-25). The dstd/at_floor/spike
            # tripwires were ALL blind to the flagship divergence (max dstd 1.8 vs
            # thresh 3.0; max at_floor 1% vs 20%; spike peaked 0.152 vs 0.20), so
            # it burned 11k steps (~18h) with the watchdog armed. This one watches
            # the actual objective. Deliberately at eval scope, not inside the
            # gate_stats block below -- that block is skipped on non-delta models.
            _vp_line, _vp_trip = _valppl_watchdog_step(_WATCHDOG, full, step, c)
            if _vp_line:
                print(_vp_line, flush=True)
            if _vp_trip:
                print(f"  [WATCHDOG] TRIPPED: val_ppl {full:.2f} exceeded best "
                      f"{_WATCHDOG['best_valppl']:.2f} (step {_WATCHDOG['best_step']}) "
                      f"for {_WATCHDOG['valppl_hits']} consecutive evals "
                      f"— saving ckpt and exiting 2", flush=True)
                _save_ckpt(ckdir, raw_model, opts, step, tokens, c, async_save=False)
                sys.exit(2)
            # β collapse detector: prints per-block β mean/dead-fraction so a
            # scalar-β write-gate saturation is visible in the training log,
            # not only in post-hoc diagnostics. dead_frac >= 0.95 on all blocks
            # = full collapse (memory-Δ will be 0). >= 0.50 on some = partial.
            # Under beta_floor > 0 the raw dead_frac is 0 by construction; the
            # collapse *pressure* signal is at_floor_frac (frac where the pre-
            # floor sigmoid is saturated to 0). Same 0.95 rule applies to it.
            beta_stats = [getattr(b, "_last_beta_stats", None) for b in raw_model.blocks]
            if any(s is not None for s in beta_stats) and c.get("delta_rule", False):
                def _fmt_beta(i, s):
                    if s is None:
                        return f"blk{i}: n/a"
                    parts = [f"blk{i}: μ={s['mean']:.2f} dead={s['dead_frac']*100:.0f}%"]
                    at_floor = s.get("at_floor_frac")
                    if at_floor is not None and c.get("beta_floor", 0.0) > 0:
                        parts.append(f"at_floor={at_floor*100:.0f}%")
                    collapsed = s["dead_frac"] >= 0.95 or (at_floor is not None and at_floor >= 0.95)
                    if collapsed:
                        parts.append("COLLAPSED")
                    return " ".join(parts)
                per_block = " | ".join(_fmt_beta(i, s) for i, s in enumerate(beta_stats))
                print(f"  [beta] {per_block}", flush=True)
            # Erase/decay-gate saturation (added 2026-08-04). The [beta] line
            # above covers only the WRITE gate; these are the two gates that
            # were unmonitored while the Step-3 run diverged.
            gate_stats = [getattr(b, "_last_gate_stats", None) for b in raw_model.blocks]
            if any(s is not None for s in gate_stats):
                def _fmt_gate(i, s):
                    if s is None:
                        return f"blk{i}: n/a"
                    lo, hi = s.get("decay_sat_lo"), s.get("decay_sat_hi")
                    d = (f"d={lo*100:.0f}/{hi*100:.0f}%" if lo is not None
                         else f"d={s['decay_sat']*100:.0f}%")
                    return (f"blk{i}: e={s['erase_sat']*100:.0f}% "
                            f"w={s['write_sat']*100:.0f}% {d} "
                            f"dstd={s['decay_std']:.1f}")
                print("  [gates sat] (d=neverforget/fastforget) " + " | ".join(
                    _fmt_gate(i, s) for i, s in enumerate(gate_stats)), flush=True)
                # Spike-dilution log (gotcha #5): attention layers push zeros
                # into spike_accum, so reported spike rate = (n_spiking/n_total)
                # * true spiking-layer rate. Report corrected number ALONGSIDE
                # (no behaviour change; sparsity penalty still uses diluted mean
                # for continuity with arm B's tuning).
                n_total = len(gate_stats)
                n_spiking = sum(1 for s in gate_stats if s is not None)
                if sr is not None and 0 < n_spiking < n_total:
                    # 2026-09-30: with spike_rate_undiluted the reported rate is ALREADY spiking-only; rescaling it
                    # doubled the logged number (signed: logged 0.264, measured 0.129; temp/diag_spike_levels.py).
                    _sp_only = sr if c.get("spike_rate_undiluted", False) else sr * n_total / n_spiking
                    print(f"  [spike] reported={sr:.3f} spiking_only={_sp_only:.3f} "
                          f"({n_spiking}/{n_total} blocks spike)", flush=True)
                # Watchdog (post-fix stability guard from docs/plan_2026_08_05.md P3).
                # Two consecutive tripwire hits → save ckpt and exit 2. Chain
                # scripts use `;` so the next arm still launches.
                if c.get("watchdog_enabled", True) and step > c.get("watchdog_grace_steps", 1500):
                    max_dstd = max((s["decay_std"] for s in gate_stats if s is not None), default=0.0)
                    dstd_thresh = c.get("watchdog_dstd_thresh", 3.0)
                    _WATCHDOG["dstd_hits"] = _WATCHDOG["dstd_hits"] + 1 if max_dstd > dstd_thresh else 0
                    max_atfloor = 0.0
                    if beta_stats:
                        max_atfloor = max((s.get("at_floor_frac") or 0.0
                                            for s in beta_stats if s is not None), default=0.0)
                    _WATCHDOG["atfloor_hits"] = _WATCHDOG["atfloor_hits"] + 1 if max_atfloor > 0.20 else 0
                    if sr is not None:
                        prev = _WATCHDOG["prev_spike"]
                        rising_past_thresh = prev is not None and sr > prev and sr > 0.20
                        _WATCHDOG["spike_hits"] = _WATCHDOG["spike_hits"] + 1 if rising_past_thresh else 0
                        _WATCHDOG["prev_spike"] = sr
                    tripped = []
                    if _WATCHDOG["dstd_hits"] >= 2:
                        tripped.append(f"dstd>{dstd_thresh:.1f} (max={max_dstd:.1f}) x{_WATCHDOG['dstd_hits']}")
                    if _WATCHDOG["atfloor_hits"] >= 2:
                        tripped.append(f"at_floor>20% (max={max_atfloor*100:.0f}%) x{_WATCHDOG['atfloor_hits']}")
                    if _WATCHDOG["spike_hits"] >= 2:
                        tripped.append(f"spike_rising past 0.20 (now {sr:.3f}) x{_WATCHDOG['spike_hits']}")
                    if tripped:
                        print(f"  [WATCHDOG] TRIPPED: {'; '.join(tripped)} — saving ckpt and exiting 2",
                              flush=True)
                        _save_ckpt(ckdir, raw_model, opts, step, tokens, c, async_save=False)
                        sys.exit(2)
            # α (decay-gate) diagnostic: sigmoid clamped to (0.5, 0.9999). Values
            # riding either clamp boundary have zero gradient there and freeze;
            # hi_frac at 0.9999 = "never forget" (memory can accumulate unboundedly);
            # lo_frac at 0.5 = "fast forget" (delta rule can't build long-range
            # bindings). Same class of silent failure as β collapse. Only prints
            # when the decay_gate arm is active AND populated the stats.
            alpha_stats = [getattr(b, "_last_alpha_stats", None) for b in raw_model.blocks]
            if any(s is not None for s in alpha_stats) and c.get("decay_gate", False):
                per_block = " | ".join(
                    (f"blk{i}: μ={s['mean']:.3f} hi={s['hi_frac']*100:.0f}% lo={s['lo_frac']*100:.0f}%"
                     + (" FROZEN_HI" if s["hi_frac"] >= 0.95 else "")
                     + (" FROZEN_LO" if s["lo_frac"] >= 0.95 else "")) if s is not None
                    else f"blk{i}: n/a"
                    for i, s in enumerate(alpha_stats))
                print(f"  [alpha] {per_block}", flush=True)
            if c.get("n_experts", 1) > 1 and raw_model._last_expert_util is not None:
                util_str = " | ".join(
                    "[" + ", ".join(f"{v:.2f}" for v in u.tolist()) + "]"
                    for u in raw_model._last_expert_util)
                peak = torch.stack(raw_model._last_route_peak).mean().item()
                print(f"  [moe] expert_util (per layer): {util_str} | avg_peak_weight {peak:.3f}",
                      flush=True)
            if c.get("ponder_enabled", False) and raw_model._last_expected_steps is not None:
                # eval runs the causal threshold path; depth_hist = fraction of
                # tokens exiting at each depth 0..S-1, keep = per-step keep-rate
                # (should track the capacity schedule if the router is calibrated).
                hist = raw_model.ponder._last_depth_hist
                hist_str = ("[" + ", ".join(f"{v:.2f}" for v in hist.tolist()) + "]"
                            if hist is not None else "n/a")
                keep = raw_model.ponder._last_eval_keep
                keep_str = ("[" + ", ".join(f"{v:.2f}" for v in keep.tolist()) + "]"
                            if keep is not None else "n/a")
                alpha = raw_model.ponder._last_alpha
                alpha_str = ("[" + ", ".join(f"{v:.3f}" for v in alpha.tolist()) + "]"
                            if alpha is not None else "n/a")
                print(f"  [ponder] avg_depth {raw_model._last_expected_steps.item():.3f}"
                      f" / max {c.get('ponder_steps', 3) - 1} | depth_hist {hist_str}"
                      f" | keep_rate {keep_str} | alpha {alpha_str}", flush=True)
            if c.get("recurrent_router_enabled", False) and raw_model._last_rdepth_depth is not None:
                # eval runs the causal threshold path; depth_hist = fraction of
                # tokens exiting at depth 1..R, keep = per-boundary keep-rate
                # (should track the capacity schedule if the router is calibrated).
                rr = raw_model.rdepth_router
                hist_str = ("[" + ", ".join(f"{v:.2f}" for v in rr._last_depth_hist.tolist()) + "]"
                            if rr._last_depth_hist is not None else "n/a")
                keep_str = ("[" + ", ".join(f"{v:.2f}" for v in rr._last_eval_keep.tolist()) + "]"
                            if rr._last_eval_keep is not None else "n/a")
                print(f"  [rdepth] avg_depth {raw_model._last_rdepth_depth.item():.3f}"
                      f" / max {c['recurrent_depth']} | depth_hist {hist_str}"
                      f" | keep_rate {keep_str}", flush=True)
            # Brain snapshot for spiking models (cheap, useful for after-the-fact analysis)
            if c["arch"] == "spiking" and c.get("snapshot_brain", True):
                try:
                    bdir = f"/home/glenn/projects/neuro/models/brains/{c['run_name']}"
                    os.makedirs(bdir, exist_ok=True)
                    # Preserve the training RNG stream. data.batch() draws twice
                    # from np.random; evaluate() guards itself but this block did
                    # not, so snapshot_brain was silently shifting which training
                    # windows the run saw (2 draws per eval). Guarded 2026-08-08
                    # so the flag is a true no-op on training.
                    _brain_rng = np.random.get_state()
                    try:
                        x, _ = data.batch(1, c["block_size"], "val", device)
                    finally:
                        np.random.set_state(_brain_rng)
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
    # when max_tokens isn't a multiple of ckpt_every). Synchronous (not async):
    # this is the last save before process exit, so there's no next step to
    # overlap I/O with, and we must not exit while a background write is
    # still in flight (would leave a truncated/corrupt .pt file).
    _save_ckpt(ckdir, raw_model, opts, step, tokens, c, async_save=False)
    print(f"done: {tokens/1e6:.1f}M tok in {(time.time()-t0)/3600:.2f}h")
    for _b in _attn_blocks: _b.attn_window = 0
    if c["arch"] == "spiking":
        _, _, sr = evaluate(raw_model, data, c, device)
        # `n_layer` (singular) is the TinyTransformer key; spiking configs use
        # `n_layers`. The old `c.get("n_layer", 2)` therefore ALWAYS missed and
        # silently benchmarked our 4-8 layer spiking models against a 2-LAYER
        # transformer, understating the energy advantage ~2x (276M printed
        # 0.82x "spiking loses"; the matched-depth number is 1.99x).
        energy.compare(model, sr, baseline_d=c["d"],
                       baseline_layers=c.get("n_layer", c.get("n_layers", 2)),
                       vocab=c["vocab"], seq_lens=[c["block_size"], 512, 4096])
    if _interrupted["hit"]:
        sys.exit(130)   # matches `timeout --signal=INT`'s conventional exit code


if __name__ == "__main__":
    main()
