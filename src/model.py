"""Phase 1+: Spiking Hebbian language model.

Token model where:
  - Leaky integrate-and-fire (LIF) layers emit sparse binary spikes (the "neurons")
  - Decaying Hebbian fast-weight matrices M store outer-product associations (the "synapses")
  - M binds the PREVIOUS token's key to the CURRENT token's value, so a later
    occurrence of a token can retrieve its learned successor (induction / recall).

      M_t = lambda * M_{t-1} + eta * (v_t  outer  k_{t-1})
      r_t = M_t q_t

Multi-layer: n_layers blocks are stacked with residual connections (requires d == d_mem
for n_layers > 1). Each layer has its own LIF state, Hebbian memory M, and optionally
SET recurrent connections.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from snntorch import Leaky, surrogate


# GDN-2 / OSDN grad-checkpointing chunk size. The GDN-2/OSDN sequential
# memory loops persist a (B, d_v, d_k) M state per timestep for BPTT, so at
# d_mem=1024, T=512, 2 layers the naive tape hits ~50 GB. Checkpointing per
# chunk of size CHUNK cuts peak activation memory ~T/CHUNK-x for a ~2x compute
# hit (one recompute per chunk during backward). At CHUNK=32,
# chunk*B*d^2*4 fits comfortably: 32*32*1024*1024*4 ≈ 4 GB per tracked tensor,
# well under the 5090's 32 GB even at batch=32. 2026-07-15: bumped 32->64 to
# halve the number of sequential Python-loop/kernel-launch iterations (GDN-2
# 90M de-risk was launch-bound at 50% GPU util despite 28GB/32GB VRAM used at
# batch=32/chunk=32). Doubles the per-tracked-tensor footprint to ~8 GB at
# batch=32 -- likely to OOM at that batch size; drop batch_size (~16-24) when
# relaunching with this change. Numerics are unaffected: checkpointing only
# changes the backward recompute schedule, not the forward math -- chunk size
# is bit-exact-preserving (verified for the analogous SDM chunk refactor in
# temp/test_sdm.py).
_GDN2_OSDN_CHUNK = 64


def _gdn2_chunk_fwd(M, k_ch, v_ch, q_ch, b_ch, w_ch, alpha_ch):
    """Per-chunk sequential loop for GDN-2. Returns (M_final, r_chunk).
    Pure function of its tensor inputs — safe target for
    torch.utils.checkpoint.checkpoint(use_reentrant=False)."""
    r_list = []
    for t in range(k_ch.size(1)):
        kt = k_ch[:, t]
        kn = kt / (kt.norm(dim=-1, keepdim=True) + 1e-6)
        a = alpha_ch[:, t].unsqueeze(1)                          # (B,1,dk)
        M_dec = M * a
        bk = (b_ch[:, t] * kn).unsqueeze(2)
        r_er = torch.bmm(M_dec, bk).squeeze(2)
        u = w_ch[:, t] * v_ch[:, t] - r_er
        M = M_dec + torch.bmm(u.unsqueeze(2), kn.unsqueeze(1))
        r_list.append(torch.bmm(M, q_ch[:, t].unsqueeze(2)).squeeze(2))
    return M, torch.stack(r_list, dim=1)


def _osdn_chunk_fwd(M, d, k_ch, v_ch, q_ch, beta_ch, alpha_ch,
                    eta, d_min, d_max):
    """Per-chunk sequential loop for OSDN. Returns (M_final, d_final, r_chunk).
    d is the online diagonal preconditioner (carried across chunks). alpha_ch
    is (B,C,1) or None (use scalar lam externally instead — the None case is
    handled by the caller by passing an ones-tensor scaled by lam)."""
    r_list = []
    for t in range(k_ch.size(1)):
        kt = k_ch[:, t]
        kn = kt / (kt.norm(dim=-1, keepdim=True) + 1e-6)
        beta = beta_ch[:, t]                                     # (B,1)
        a = alpha_ch[:, t].unsqueeze(2)                          # (B,1,1)
        prevM = a * M
        Mk = torch.bmm(prevM, kn.unsqueeze(2)).squeeze(2)
        u = beta * (v_ch[:, t] - Mk)
        k_tilde = d * kn
        M = prevM + torch.bmm(u.unsqueeze(2), k_tilde.unsqueeze(1))
        r_list.append(torch.bmm(M, q_ch[:, t].unsqueeze(2)).squeeze(2))
        k2 = kn * kn
        nrm = k2.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        dot = (d * k2).sum(dim=-1, keepdim=True)
        d = (d + (eta * beta / nrm) * (1 - beta * dot) * k2).clamp(d_min, d_max)
    return M, d, torch.stack(r_list, dim=1)


class RMSNorm(nn.Module):
    """Root-mean-square normalization (Llama / DeepSeek convention).
    Cheaper than LayerNorm (no mean subtraction, no bias), often slightly more
    stable. Drop-in replacement; state-dict-incompatible with LayerNorm so
    only swap on fresh runs."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        inv = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * inv).to(x.dtype) * self.weight


def _make_norm(d, use_rmsnorm=False):
    return RMSNorm(d) if use_rmsnorm else nn.LayerNorm(d)


@torch.compiler.disable
@torch.no_grad()
def _gate_sat_stats(b_logit, w_logit, f_logit, sat=4.595):
    """Saturation telemetry for all three GDN-2 gates.

    |logit| > 4.595 <=> sigmoid past 1% of its rail, where the gradient is
    ~0 and the gate can no longer adapt. Only the write gate was ever logged
    before; that blindness hid the actual Step-3 failure (blocks 0-2 decay
    gates blown up to std 6-7, blocks 4-7 erase gates 58-92% saturated).

    Reports each gate's saturated fraction plus the decay-logit std, which is
    the earliest and clearest signal — healthy blocks sit at 2.2-2.9, blocks
    that went binary hit 6.2-7.3.
    """
    def sat_frac(t):
        return float((t.abs() > sat).float().mean().item())
    return {
        "erase_sat": sat_frac(b_logit),
        "write_sat": sat_frac(w_logit),
        "decay_sat": sat_frac(f_logit),
        # Signed split (2026-08-08). |logit|>sat conflates two OPPOSITE
        # pathologies and the aggregate cannot tell them apart:
        #   f << 0  -> softplus~0 -> alpha~1  -> channel never forgets
        #   f >> 0  -> softplus~f -> alpha~0  -> channel wiped every token
        # Measured on p4_seed0 the saturation is 100% one-sided (blk0 35.4%
        # never-forget / 0.1% fast-forget), which the single number hid.
        "decay_sat_lo": float((f_logit < -sat).float().mean().item()),
        "decay_sat_hi": float((f_logit > sat).float().mean().item()),
        "decay_std": float(f_logit.float().std().item()),
    }


@torch.compiler.disable
@torch.no_grad()
def _beta_stats(beta_seq, floor=0.0):
    """Small per-block β diagnostic for scalar-β collapse detection.

    Records mean/min/max/dead_frac (fraction of β < 0.01, i.e. sigmoid
    saturated) so the trainer can print it during eval and catch the
    write-gate collapse pathology *while training*, instead of finding
    it post-hoc after hours of runtime. Cheap: ~5 reductions per block
    per forward, always-on, no config flag.

    at_floor_frac: under beta_floor > 0 the dead threshold (β < 0.01) can
    never trip -- β is >= floor by construction. Recover the collapse
    *pressure* signal by inverting the floor to get the pre-floor sigmoid
    value σ = (β - floor)/(1 - floor), and count σ < 0.01 as "at floor".
    Same interpretation as dead_frac when floor=0. This is what the
    project's "watch dead_frac climbing" rule actually needs to observe
    once beta_floor: 0.1 became the recipe default (see journey §33/§36).

    Read by train_gpu evaluate() via block._last_beta_stats.
    """
    b = beta_seq.detach()
    if floor > 0.0:
        pre = ((b - floor) / max(1e-6, 1.0 - floor)).clamp_min(0.0)
    else:
        pre = b
    return {
        "mean":          float(b.mean().item()),
        "min":           float(b.min().item()),
        "max":           float(b.max().item()),
        "dead_frac":     float((b < 0.01).float().mean().item()),
        "at_floor_frac": float((pre < 0.01).float().mean().item()),
    }


@torch.compiler.disable
@torch.no_grad()
def _alpha_stats(alpha_seq, lo=0.5, hi=0.9999):
    """α (decay-gate) diagnostic. Analogous silent failure mode to β collapse:
    sigmoid(W_α)·clamp(lo, hi) can saturate at either boundary, where the
    clamp zeroes the gradient and the gate freezes. `hi_frac` at the never-
    forget boundary is the most dangerous (memory grows unbounded / no
    contextual forgetting); `lo_frac` at the fast-forget boundary means
    memory is being erased every step and the delta rule can't accumulate.
    """
    a = alpha_seq.detach()
    return {
        "mean":    float(a.mean().item()),
        "hi_frac": float((a > (hi - 1e-4)).float().mean().item()),
        "lo_frac": float((a < (lo + 1e-4)).float().mean().item()),
    }


class LowRankLinear(nn.Module):
    """Rank-r bottleneck drop-in for nn.Linear(in, out): y = up(down(x)).

    Exposes `.weight` and `.bias` as properties onto the UP factor so every
    existing call site and initializer keeps working untouched -- including
    `W_fdecay.weight.mul_(0.01)` (scaling the up factor scales the product by the
    same amount) and `W_fdecay.bias.fill_(-3.0)`.

    Both factors are ordinary 2D weights with min(shape) = r > 1, so muon.py
    classifies them exactly as it classifies the full gates today (Muon under
    precise_gate_filter) with no routing change. They are NOT LoRA adapters on a
    frozen base -- this is the gate's only parameterization -- so Newton-Schulz
    is as appropriate here as on any hidden matrix.
    """

    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.rank = rank
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)

    @property
    def weight(self):
        return self.up.weight

    @property
    def bias(self):
        return self.up.bias

    def forward(self, x):
        return self.up(self.down(x))


class SwiGLUFF(nn.Module):
    """SwiGLU gated feed-forward (Llama / DeepSeek convention).
    Replaces a single Linear(d_in, d_out) with:
        down(SiLU(gate(x)) * up(x))
    Adds ~3x the params of a single Linear when hidden=d_out. Trades that for
    a consistent 2-5% val_ppl lift across LLM literature."""
    def __init__(self, d_in, d_out, hidden=None, bias=False):
        super().__init__()
        h = hidden or d_out
        self.gate = nn.Linear(d_in, h, bias=bias)
        self.up   = nn.Linear(d_in, h, bias=bias)
        self.down = nn.Linear(h, d_out, bias=bias)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SpikingHebbianBlock(nn.Module):
    """Single spiking-Hebbian layer."""

    def __init__(self, d_in, n_neurons, d_mem, beta=0.9, lam=0.98, eta=1.0,
                 compile_safe=False, recurrent=False, rec_density=0.05,
                 learnable_decay=False, write_gate=False, delta_rule=False,
                 beta_floor=0.0, decay_gate=False, titans=False,
                 local_attn=False, local_window=64,
                 n_heads=1, pre_conv=False, pre_conv_kernel=4,
                 vector_beta=False, use_rmsnorm=False,
                 use_swiglu=False, depth_scale=1.0, compile_inv=False,
                 use_fla_kernel=False, gdn2_enabled=False, osdn_enabled=False,
                 osdn_eta=0.003, osdn_dmin=0.5, osdn_dmax=2.0,
                 sdm_enabled=False, sdm_N=None, sdm_W=4, sdm_R=4,
                 gdn2_min_heads=1, gdn2_chunked=True, gdn2_ckpt_chunks=True,
                 gdn2_chunk_size=64, gdn2_gate_norm=False,
                 gdn2_loglinear=False, gdn2_loglinear_L=None, deltaproduct_nh=1,
                 gate_rank=0, lif_exact=False, lif_reset=1.0):
        super().__init__()
        self.n_neurons = n_neurons
        # lif_exact: exact sequential LIF Triton kernel (src/lif_exact.py) instead of
        # the K-capped FPT lif_parallel (which never converges at T=512). Default off.
        self.lif_exact = lif_exact
        self.lif_reset = lif_reset   # lif_exact only: reset strength gamma (1=LIF, 0=no-reset PSN)
        self.d_mem = d_mem
        self.n_heads = n_heads
        self.deltaproduct_nh = deltaproduct_nh
        # Whether to grad-checkpoint each GDN-2 dense chunk during training.
        # True (default) matches every screened run to date; False saves the
        # ~40% chunk-recompute at the cost of ~10-20 GB extra activation tape
        # per layer at batch=15/block=512 (per Fable-on-High audit 2026-07-16).
        # Numerics unaffected either way (checkpointing only changes the
        # backward recompute schedule; forward math is identical).
        self.gdn2_ckpt_chunks = gdn2_ckpt_chunks
        # Sequential chunk size for gdn2_chunked_dense (delta_chunked.py). 64
        # was picked as a launch-bound fix at 90M/4-layers (2026-07-15, see
        # _GDN2_OSDN_CHUNK comment above); deeper stacks re-hit the same
        # launch-bound wall (more layers = more sequential Python-loop chunk
        # iterations). Exact reassociation of the same math at any chunk size
        # (parity-tested); only float-order and speed change.
        self.gdn2_chunk_size = gdn2_chunk_size
        # GDN-2 gate telemetry only in eval mode (set by the LM from config).
        self.telemetry_eval_only = False
        if d_mem % n_heads != 0:
            raise ValueError(f"d_mem ({d_mem}) must be divisible by n_heads ({n_heads})")
        self.d_head = d_mem // n_heads
        # LM-level state allocator reads these to size the carried M-state.
        self.n_state_units = n_heads
        self.d_state = self.d_head
        self.lam = lam
        self.eta = eta
        self.compile_safe = compile_safe
        self.beta_val = beta
        self.learnable_decay = learnable_decay
        self.write_gate = write_gate
        self.delta_rule = delta_rule
        self.beta_floor = beta_floor   # min write strength: β = floor + (1-floor)·σ(W_β)
        self.decay_gate = decay_gate   # Gated DeltaNet: data-dependent forget α = σ(W_alpha(spk))
        # ST Phase 8: delta-rule write strength β = sigmoid(W_beta(spk)).
        # vector_beta (RWKV-7 style): per-channel β instead of scalar — write
        # strength can differ across memory-output dimensions.
        self.vector_beta = vector_beta
        if delta_rule:
            self.W_beta = nn.Linear(n_neurons, d_mem if vector_beta else 1)
        # GDN-2 (arXiv:2605.22791) and OSDN (arXiv:2605.13473): delta-rule recall
        # upgrades — see docs/gdn2_osdn_design.md. Sequential-loop kernel only for
        # now (chunked WY deferred until a screen wins). Both require delta_rule.
        self.gdn2_enabled = gdn2_enabled
        self.osdn_enabled = osdn_enabled
        # Diagnostic-only override: force the fla-kernel GDN-2 wrapper to split
        # into at least this many heads (it auto-picks the minimum satisfying
        # the kernel's 256 key-headdim cap; this forces MORE fragmentation than
        # the minimum, for charting the recall-vs-head-count curve). 1 = no-op.
        self.gdn2_min_heads = gdn2_min_heads
        # Dense chunked-WY fast path (delta_chunked.gdn2_chunked_dense):
        # exact same math as the reference loop (parity ~1e-6, incl. grads),
        # ~26x faster. False falls back to the eager per-token loop.
        self.gdn2_chunked = gdn2_chunked
        if gdn2_enabled or osdn_enabled:
            if not delta_rule:
                raise ValueError("gdn2_enabled/osdn_enabled require delta_rule: true")
            if n_heads > 1 or vector_beta or titans:
                raise ValueError("gdn2/osdn incompatible with n_heads>1, vector_beta, titans")
            if gdn2_enabled and osdn_enabled:
                raise ValueError("gdn2_enabled and osdn_enabled are mutually exclusive")
        if gdn2_enabled:
            # Channel-wise erase/write/decay gates replace the scalar W_beta/W_alpha.
            # w_t is floored (reuses beta_floor) — carries the β-collapse lesson.
            # gate_rank r > 0 replaces each full (n_neurons -> d_mem) gate with a
            # rank-r bottleneck. MEASURED (scripts/mfu_report.py on
            # flagship_v3_363m): these three gates are 50.2% of ALL training
            # FLOPs -- W_fdecay 19.3%, W_w 16.7%, W_b 14.2% -- while the whole
            # attention layer is 1.8%. At n_neurons=3840/d_mem=1280 each gate is
            # 4.92M params; at r=32 it is 164k, 30x cheaper.
            # Precedent: GLA (arXiv:2312.06635) parameterizes its forget gate as
            # exactly this kind of bottleneck at RANK 16, explicitly because the
            # full mapping is "prohibitively expensive", and ablates it as
            # sufficient (ppl 14.77 rank-16 vs 15.56 data-independent scalar
            # decay vs 23.21 no gate) -- the gate must be data-dependent, but it
            # does NOT need full rank.
            # gate_rank=0 (default) builds the original nn.Linear -> byte-identical.
            _gate = (lambda: nn.Linear(n_neurons, d_mem)) if not gate_rank else (
                lambda: LowRankLinear(n_neurons, d_mem, gate_rank))
            self.W_b = _gate()                                     # erase gate (key dim)
            self.W_w = _gate()                                     # write gate (value dim)
            self.W_fdecay = _gate()                                # channel-wise log-decay
            self.gdn2_a = nn.Parameter(torch.zeros(d_mem))         # decay scale (exp(a)=1 init)
            with torch.no_grad():
                # softplus(bias) small & positive so α = exp(−exp(a)·softplus)≈1
                # at init (slow decay, matches default lam≈0.98 behavior).
                self.W_fdecay.weight.mul_(0.01)
                self.W_fdecay.bias.fill_(-3.0)                     # softplus(-3)≈0.049
            # Gate-logit normalization (2026-08-04). The gate logit is a sum over
            # ~n_neurons*spike_rate ACTIVE spike units, so once W_b/W_w/W_fdecay
            # learn a coherent direction the logit magnitude grows with WIDTH.
            # At n_neurons=1536 it stays inside the sigmoid's useful band; at
            # 2752 it saturates (measured: blk1 write 46% saturated, blk4-7
            # erase 58-92%, blk0-2 decay-logit std 6.2-7.3), gradient -> 0, and
            # the gate can no longer adapt. RMSNorm divides that growth out.
            # See temp/diag_gate_distributions.py and the Step-3 divergence memo.
            self.gdn2_gate_norm = gdn2_gate_norm
            if gdn2_gate_norm:
                self.gate_norm_b = RMSNorm(d_mem)
                self.gate_norm_w = RMSNorm(d_mem)
                self.gate_norm_f = RMSNorm(d_mem)
                # RMSNorm would wash out W_fdecay's deliberate bias=-3.0
                # slow-decay init, so carry that offset in an explicit
                # post-norm bias instead (same effective init).
                self.gate_bias_f = nn.Parameter(torch.full((d_mem,), -3.0))
            # Log-linear GDN-2 (arXiv:2506.04761, Stage-A spike; design doc
            # docs/loglinear_attention_design.md, memory
            # neuro_loglinear_stageA_kernel_2026_08_28). Chunk-level Fenwick
            # tower of completed-chunk states, read-reweighted by a tiny per-
            # step head; write path is UNCHANGED. L is the number of tower
            # levels (default: log2(max_block/chunk)+2). W_lw emits (L+1) per-
            # step scalars: L level weights + 1 base weight, sigmoid-scaled to
            # (0, 2) so init ≈ 1 (near-GDN-2 at init, the trainable delta being
            # the recall-ceiling lift).
            self.gdn2_loglinear = gdn2_loglinear
            if gdn2_loglinear:
                if not gdn2_chunked:
                    raise ValueError("gdn2_loglinear requires gdn2_chunked: true")
                if gdn2_loglinear_L is None:
                    # 2^L >= max n_chunks. Default assumes block_size <= 4096
                    # and gdn2_chunk_size >= 32 (Fenwick n_chunks <= 128).
                    gdn2_loglinear_L = 8
                self.gdn2_loglinear_L = gdn2_loglinear_L
                self.W_lw = nn.Linear(n_neurons, gdn2_loglinear_L + 1)
                with torch.no_grad():
                    # 2 * sigmoid(0) = 1 -- ~= plain GDN-2 at init, trainable
                    # up to weight 2 (or down to 0 = "ignore this level").
                    self.W_lw.weight.mul_(0.01)
                    self.W_lw.bias.zero_()
            # DeltaProduct (arXiv:2502.10297): n_h Householder products/token via
            # sequence expansion + the existing gdn2_chunked_dense kernel (verified
            # temp/deltaproduct_reference.py + temp/test_deltaproduct_plumbing.py).
            # n_h=1 (default) constructs nothing -> byte-identical to current runs.
            if deltaproduct_nh > 1:
                if not gdn2_chunked or use_fla_kernel or gdn2_loglinear:
                    raise ValueError('deltaproduct_nh>1 needs gdn2_chunked (not fla/loglinear)')
                from deltaproduct import DeltaProductExtra
                self.dp = DeltaProductExtra(n_neurons, d_mem, deltaproduct_nh,
                                            beta_floor, gate_norm=gdn2_gate_norm,
                                            norm_ctor=RMSNorm)
        if osdn_enabled:
            # No learned params — d_t is online-adapted state (reset per window).
            self.osdn_eta = osdn_eta
            self.osdn_dmin = osdn_dmin
            self.osdn_dmax = osdn_dmax
        # SDM (arXiv:2607.07386) — see docs/sdm_design.md. Replaces the dense
        # (d_v × d_k) delta-rule memory with an N-slot PKM-addressed bank.
        # Reuses this block's W_beta/W_alpha (β/α stay owned here so
        # beta_floor telemetry / decay_gate keep working unchanged) but the
        # k/q addressing and the write/read itself are delegated to
        # SparseSDMMemory. Phase-3 scope constraint: sdm_N must equal d_mem
        # so the state stays square (N, N) — this lets the existing LM-level
        # state allocator (which assumes a (d_state, d_state) square carried
        # state) work completely unchanged. Non-square SDM sizing (the 500M
        # restart plan's N=16384 with d_mem=2048) needs a state-allocator
        # refactor (per-block init_state()), deferred — not required for the
        # 30M screen, whose design-doc sizing already specifies N=1024=d_mem.
        self.sdm_enabled = sdm_enabled
        if sdm_enabled:
            if not delta_rule:
                raise ValueError("sdm_enabled requires delta_rule: true")
            if gdn2_enabled or osdn_enabled or vector_beta or titans:
                raise ValueError("sdm_enabled incompatible with gdn2/osdn/vector_beta/titans")
            if n_heads > 1:
                raise ValueError("sdm_enabled incompatible with n_heads>1 (phase 3 scope)")
            sdm_N = sdm_N or d_mem
            if sdm_N != d_mem:
                raise ValueError(
                    f"sdm_enabled currently requires sdm_N == d_mem (got sdm_N={sdm_N}, "
                    f"d_mem={d_mem}) -- non-square SDM sizing needs a state-allocator "
                    f"refactor (see docs/sdm_design.md phase 3 note), not yet built")
            from sdm import SparseSDMMemory
            self.sdm = SparseSDMMemory(n_neurons, sdm_N, d_mem, W=sdm_W, R=sdm_R)
        self.sdm_N = sdm_N
        if decay_gate:
            # Init bias high so initial α ≈ 0.98 (matches default lam) — model can
            # then learn to LOWER α (forget faster) where useful.
            self.W_alpha = nn.Linear(n_neurons, 1)
            with torch.no_grad():
                self.W_alpha.weight.mul_(0.01)
                self.W_alpha.bias.fill_(4.0)        # sigmoid(4.0) ≈ 0.982
        self.titans = titans
        if titans:
            # Titans: gradient-descent memory with momentum + data-dependent forget.
            # θ (write strength / learning rate), η (momentum decay), α (forget gate).
            # Init: θ small (gentle write), η high (preserve momentum), α small (mostly keep).
            self.W_theta = nn.Linear(n_neurons, 1)
            self.W_eta   = nn.Linear(n_neurons, 1)
            self.W_alpha_t = nn.Linear(n_neurons, 1)
            with torch.no_grad():
                for L_, b in ((self.W_theta, -2.0), (self.W_eta, 2.0), (self.W_alpha_t, -4.0)):
                    L_.weight.mul_(0.01); L_.bias.fill_(b)
            # Titans also needs k,v,q projections (already created below).
        self.local_attn = local_attn
        self.local_window = local_window
        if local_attn:
            # BASED-style: tiny causal sliding-window exact attention added to the
            # memory read. Reuses W_k/W_v/W_q. A learnable per-block scalar gate so
            # the model can pick how much local-recall help it wants (init 0 -> no-op
            # at training start; weights gain ground gradually).
            self.local_gate = nn.Parameter(torch.zeros(1))
        self.pre_conv = pre_conv
        self.pre_conv_kernel = pre_conv_kernel
        if pre_conv:
            # Per 2508.19029: causal depthwise 1D conv on spike sequence before
            # the W_k/W_v/W_q projections. Improves recall (mixes neighbor timesteps
            # so the projection sees a local window). Depthwise (groups=N) = cheap.
            self.spk_conv = nn.Conv1d(n_neurons, n_neurons,
                                       kernel_size=pre_conv_kernel,
                                       padding=0, groups=n_neurons)

        # ST Phase 4: per-(value-dim) learnable decay. alpha = sigmoid(raw).
        # Init MULTI-SCALE (RetNet-style): spread alpha across dims so the memory
        # has many timescales from the start and each dim gets distinct gradient
        # signal (the clumped alpha=0.99 init never differentiated). 1-alpha is
        # geometric from 0.001 (alpha 0.999, long memory) to 0.10 (alpha 0.90, short).
        if learnable_decay:
            one_minus = torch.logspace(math.log10(0.001), math.log10(0.10), d_mem)
            alphas = (1.0 - one_minus).clamp(0.5, 0.9999)
            self.decay_raw = nn.Parameter(torch.logit(alphas))
        # ST Phase 5: learnable scalar write gate g_t = sigmoid(W_gate(spk)).
        if write_gate:
            self.W_gate = nn.Linear(n_neurons, 1)

        self.to_current = nn.Linear(d_in, n_neurons)
        if not compile_safe:
            self.lif = Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        if recurrent:
            self.W_rec = nn.Linear(n_neurons, n_neurons, bias=False)
            mask = (torch.rand(n_neurons, n_neurons) < rec_density).float()
            mask.fill_diagonal_(0)
            self.register_buffer("rec_mask", mask)
        self.recurrent = recurrent
        self.compile_inv = compile_inv
        self.use_fla_kernel = use_fla_kernel
        self.W_k = nn.Linear(n_neurons, d_mem, bias=False)
        self.W_v = nn.Linear(n_neurons, d_mem, bias=False)
        self.W_q = nn.Linear(n_neurons, d_mem, bias=False)
        # Batch B: SwiGLU FFN. Same input/output dims as the original Linear
        # so all downstream shape assumptions hold; ~3x the FFN params.
        self.use_swiglu = use_swiglu
        if use_swiglu:
            self.W_ff = SwiGLUFF(n_neurons, d_mem, bias=False)
        else:
            self.W_ff = nn.Linear(n_neurons, d_mem, bias=False)
        self.norm = _make_norm(d_mem, use_rmsnorm)

        # Batch B: depth-scaled init (GPT-NeoX / DeepSeek convention).
        # Scale the output projection weight — the last Linear before the
        # residual — by 1/sqrt(2*n_layers) to keep residual-stream variance
        # bounded as depth grows. depth_scale is the FACTOR (already computed
        # by the LM wrapper) so we can just multiply.
        if depth_scale != 1.0:
            out_lin = self.W_ff.down if use_swiglu else self.W_ff
            with torch.no_grad():
                out_lin.weight.mul_(depth_scale)

    def _lif_step(self, cur, mem):
        """Inline LIF (vth=1, reset-by-subtraction, atan surrogate). Compile-friendly."""
        mem = self.beta_val * mem + cur
        over = mem - 1.0
        spk_hard = (over > 0).float()
        alpha = 2.0
        g = (1.0 / (math.pi * alpha)) * torch.atan(math.pi * alpha * over)
        spk = spk_hard.detach() + g - g.detach()
        mem = mem - spk_hard
        return spk, mem

    def step(self, x, mem, M, prev_spk, ablate_memory=False):
        """Process one timestep. Returns (output, mem, M, spk)."""
        if self.gdn2_enabled or self.osdn_enabled:
            raise NotImplementedError(
                "gdn2/osdn have no token-by-token step() path yet; screen via "
                "forward_sequence (use_fpt: true). See docs/gdn2_osdn_design.md.")
        if self.sdm_enabled:
            raise NotImplementedError(
                "sdm has no token-by-token step() path yet; screen via "
                "forward_sequence (use_fpt: true). See docs/sdm_design.md.")
        cur = self.to_current(x)
        if self.recurrent:
            cur = cur + F.linear(prev_spk, self.W_rec.weight * self.rec_mask)
        if self.compile_safe:
            spk, mem = self._lif_step(cur, mem)
        else:
            spk, mem = self.lif(cur, mem)
        k = self.W_k(prev_spk)
        v = self.W_v(spk)
        q = self.W_q(spk)
        ff = self.W_ff(spk)
        if self.write_gate:
            v = v * torch.sigmoid(self.W_gate(spk))          # selective write
        if not ablate_memory:
            if self.learnable_decay:
                a = torch.sigmoid(self.decay_raw).clamp(0.5, 0.9999).view(1, -1, 1)
            if self.delta_rule:
                # Gated DeltaNet: M = αM + β(v − αM·k̂)⊗k̂  (error-driven write).
                # k̂ = L2-normalized key; β = learned input-dependent write strength
                # (scalar (B,1) or per-channel vector (B,D) when vector_beta=True);
                # α = learned forget gate (decay_gate=True) or scalar lam.
                kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
                beta = self.beta_floor + (1 - self.beta_floor) * torch.sigmoid(self.W_beta(spk))  # (B,1) or (B,D)
                if self.decay_gate:
                    a_dyn = torch.sigmoid(self.W_alpha(spk)).clamp(0.5, 0.9999).unsqueeze(2)
                    prevM = a_dyn * M
                elif self.learnable_decay:
                    prevM = a * M
                else:
                    prevM = self.lam * M
                Mk = torch.bmm(prevM, kn.unsqueeze(2)).squeeze(2)  # λM·k̂  (B,dm)
                delta = (beta * (v - Mk)).unsqueeze(2)            # (B,dm,1)
                M = prevM + torch.bmm(delta, kn.unsqueeze(1))
            elif self.learnable_decay:
                M = a * M + self.eta * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
            else:
                M = self.lam * M + self.eta * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
            r = torch.bmm(M, q.unsqueeze(2)).squeeze(2)
        else:
            r = torch.zeros_like(ff)
        out = self.norm(r + ff)
        return out, mem, M, spk

    def step_titans(self, x, mem, M, S, prev_spk, ablate_memory=False):
        """Titans one-step update: gradient-descent memory with momentum + forget.
            e = M k̂ − v ; S ← η S − θ e k̂ᵀ ; M ← (1−α) M + S ; r = M q
        """
        cur = self.to_current(x)
        if self.recurrent:
            cur = cur + torch.nn.functional.linear(prev_spk, self.W_rec.weight * self.rec_mask)
        if self.compile_safe:
            spk, mem = self._lif_step(cur, mem)
        else:
            spk, mem = self.lif(cur, mem)
        k = self.W_k(prev_spk); v = self.W_v(spk); q = self.W_q(spk); ff = self.W_ff(spk)
        if self.write_gate:
            v = v * torch.sigmoid(self.W_gate(spk))
        if not ablate_memory:
            kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            th = torch.sigmoid(self.W_theta(spk)).unsqueeze(2)        # (B,1,1) learning rate
            et = torch.sigmoid(self.W_eta(spk)).unsqueeze(2)          # momentum decay
            al = torch.sigmoid(self.W_alpha_t(spk)).unsqueeze(2)      # forget gate
            Mk = torch.bmm(M, kn.unsqueeze(2)).squeeze(2)
            e = Mk - v                                                 # surprise (B,D)
            grad = torch.bmm(e.unsqueeze(2), kn.unsqueeze(1))         # (B,D,D)
            S = et * S - th * grad
            M = (1 - al) * M + S
            r = torch.bmm(M, q.unsqueeze(2)).squeeze(2)
        else:
            r = torch.zeros_like(ff)
        out = self.norm(r + ff)
        return out, mem, M, S, spk

    def forward_sequence(self, x_seq, mem, M, prev_spk, ablate_memory=False,
                         fpt_K=10):
        """FPT-parallel forward over the whole sequence.

        x_seq: (B, T, d_in)
        Returns: out_seq (B, T, d_mem), mem_final (B, N), M_final (B, dm, dm),
                 spk_seq (B, T, N) for return_stats use, final_spk (B, N).

        Pipeline:
          1. to_current (parallel over T)
          2. recurrent input: in FPT mode we use prev-spike from previous BPTT
             window only for t=0; for t>=1 we approximate using the spike at t-1
             from THIS sequence -- exact under FPT once converged.
          3. LIF: lif_parallel (FPT, K iterations of parallel scan)
          4. k/v/q/ff projections (parallel)
          5. Hebbian write/read (sequential loop -- light)
        """
        if self.delta_rule and self.learnable_decay:
            raise NotImplementedError(
                "chunked delta + per-dim learnable_decay not supported; use decay_gate "
                "(per-step scalar forget) instead, or sequential step() path.")
        if self.titans:
            raise NotImplementedError(
                "titans has no chunked kernel yet; train with use_fpt=false "
                "(sequential step_titans path).")
        if self.lif_exact:
            import functools
            from lif_exact import lif_exact
            lif_parallel = functools.partial(lif_exact, gamma=self.lif_reset)
        else:
            from lif_parallel import lif_parallel
        B, T, _ = x_seq.shape
        cur_seq = self.to_current(x_seq)                       # (B, T, N)
        if self.recurrent:
            # prev_spk at t=0 from carry, at t>=1 from "previous-in-sequence".
            # Cheap approximation: use a one-step FPT preview. We use prev_spk
            # carry shifted in; the within-sequence recurrent contribution is
            # left to be picked up implicitly through the cur->spk->next-cur path
            # by re-running FPT (this is one of the K iterations).
            pad = prev_spk.unsqueeze(1)                        # (B, 1, N)
            # We don't know within-sequence prev_spk yet; use carry for t=0
            # and zeros for t>=1, then refine after spikes are known.
            zeros = torch.zeros(B, T - 1, self.n_neurons,
                                device=x_seq.device, dtype=x_seq.dtype)
            prev_seq_init = torch.cat([pad, zeros], dim=1)     # (B, T, N)
            cur_seq = cur_seq + F.linear(prev_seq_init,
                                          self.W_rec.weight * self.rec_mask)

        spk_seq, mem_final = lif_parallel(cur_seq, self.beta_val, vth=1.0,
                                          K=fpt_K)

        # If recurrent, refine once: now we know spk_seq, rebuild cur with the
        # correct prev_spk = shift_right(spk_seq, fill=carry), and re-run LIF.
        if self.recurrent:
            prev_seq = torch.cat([prev_spk.unsqueeze(1), spk_seq[:, :-1, :]],
                                 dim=1)
            cur_seq2 = self.to_current(x_seq) + F.linear(
                prev_seq, self.W_rec.weight * self.rec_mask)
            spk_seq, mem_final = lif_parallel(cur_seq2, self.beta_val, vth=1.0,
                                              K=fpt_K)

        # prev_spk shifted (k uses PREVIOUS token's spike)
        prev_seq = torch.cat([prev_spk.unsqueeze(1), spk_seq[:, :-1, :]], dim=1)
        # Pre-conv (depthwise causal 1D conv on spike sequence) before W_k/v/q.
        # FF residual uses unconvolved spk_seq (keep residual clean).
        if self.pre_conv:
            K = self.pre_conv_kernel
            spk_for_qv = F.pad(spk_seq.transpose(1, 2), (K - 1, 0))   # (B,N,T+K-1)
            spk_for_qv = self.spk_conv(spk_for_qv).transpose(1, 2)    # (B,T,N) causal
            prev_for_k = F.pad(prev_seq.transpose(1, 2), (K - 1, 0))
            prev_for_k = self.spk_conv(prev_for_k).transpose(1, 2)
        else:
            spk_for_qv = spk_seq
            prev_for_k = prev_seq
        k_seq = self.W_k(prev_for_k)                            # (B, T, dm)
        v_seq = self.W_v(spk_for_qv)
        q_seq = self.W_q(spk_for_qv)
        ff_seq = self.W_ff(spk_seq)

        # Hebbian: chunkwise-parallel via hebbian_chunked (TFLA-style).
        if self.write_gate:                                  # ST Phase 5
            v_seq = v_seq * torch.sigmoid(self.W_gate(spk_seq))
        if not ablate_memory:
            # Multi-head: split d_mem into H independent heads, collapse to batch.
            # (B,T,D) -> (B,T,H,Dh) -> (B*H, T, Dh) ; M is already (B*H, Dh, Dh).
            if self.n_heads > 1:
                H, Dh = self.n_heads, self.d_head
                v_in = v_seq.reshape(B, T, H, Dh).permute(0, 2, 1, 3).reshape(B * H, T, Dh)
                k_in = k_seq.reshape(B, T, H, Dh).permute(0, 2, 1, 3).reshape(B * H, T, Dh)
                q_in = q_seq.reshape(B, T, H, Dh).permute(0, 2, 1, 3).reshape(B * H, T, Dh)
            else:
                v_in, k_in, q_in = v_seq, k_seq, q_seq
            if self.delta_rule:                              # ST Phase 8 (chunked)
                if self.gdn2_enabled:
                    # Gated DeltaNet-2 (arXiv:2605.22791): channel-wise erase/write
                    # decoupling + channel-wise decay. Sequential memory loop with
                    # grad-checkpointed chunks (~16x peak-VRAM reduction vs one
                    # unbroken tape at T=512). LIF stays FPT-parallel.
                    b_logit = self.W_b(spk_seq)
                    w_logit = self.W_w(spk_seq)
                    f_logit = self.W_fdecay(spk_seq)
                    if self.gdn2_gate_norm:
                        b_logit = self.gate_norm_b(b_logit)
                        w_logit = self.gate_norm_w(w_logit)
                        f_logit = self.gate_norm_f(f_logit) + self.gate_bias_f
                    b_seq = torch.sigmoid(b_logit)                         # (B,T,dk) erase
                    w_seq = self.beta_floor + (1 - self.beta_floor) * torch.sigmoid(w_logit)  # write (floored)
                    alpha_seq = torch.exp(-torch.exp(self.gdn2_a)
                                          * F.softplus(f_logit))           # (B,T,dk) decay
                    # telemetry_eval_only (2026-09-25): these helpers are
                    # compiler-disabled (graph breaks) and do ~14 .item() host
                    # syncs per block per forward. The trainer only reads them
                    # after evaluate(), whose last forward is in eval mode, so
                    # skipping them in training changes no logged number.
                    if not (self.telemetry_eval_only and self.training):
                        self._last_beta_stats = _beta_stats(w_seq, floor=self.beta_floor)  # telemetry on write gate (floored)
                        # Erase/decay telemetry: for this whole investigation only the
                        # WRITE gate was logged, which hid a blown-up decay gate in
                        # block 0 and 58-92% erase saturation in blocks 4-7.
                        self._last_gate_stats = _gate_sat_stats(b_logit, w_logit, f_logit)
                        # alpha-space readout (2026-08-08). The [alpha] line was
                        # dead on this path -- only the OSDN/SDM/plain-delta
                        # branches ever populated it -- so the effective memory
                        # horizon of the GDN-2 recipe was never logged. Here
                        # alpha in (0,1) unclamped, so hi/lo bracket the raw
                        # range: hi_frac = never-forget, lo_frac = wiped.
                        self._last_alpha_stats = _alpha_stats(alpha_seq, lo=0.0, hi=1.0)
                    if self.use_fla_kernel:
                        # Fused Triton kernel — parity-validated vs the loop
                        # below (temp/test_gdn2_fla_parity.py). See
                        # delta_chunked.gdn2_fla_gated docstring for the
                        # convention mapping and the d_mem>256 block-diagonal
                        # caveat.
                        from delta_chunked import gdn2_fla_gated
                        r_in, M = gdn2_fla_gated(v_in, k_in, q_in, b_seq, w_seq, alpha_seq, M, min_heads=self.gdn2_min_heads)
                    elif self.gdn2_loglinear:
                        # Log-linear GDN-2 (Stage-A spike). Same write path as
                        # gdn2_chunked_dense (parity-anchor at lw=1,bw=1); only
                        # the READ is reweighted per Fenwick level. The tiny
                        # W_lw head produces (L+1) per-step scalars in (0,2).
                        # See docs/loglinear_attention_design.md, temp/
                        # test_loglinear_gpu.py (fwd+bwd parity), temp/
                        # stress_loglinear_nan.py (500-iter NaN-free at trained
                        # regime).
                        from loglinear import gdn2_loglinear_chunked
                        lw_all = 2.0 * torch.sigmoid(self.W_lw(spk_seq))     # (B,T,L+1)
                        lw_seq = lw_all[..., : self.gdn2_loglinear_L]        # (B,T,L)
                        bw_seq = lw_all[..., self.gdn2_loglinear_L]          # (B,T)
                        r_in, M = gdn2_loglinear_chunked(
                            v_in, k_in, q_in, b_seq, w_seq, alpha_seq,
                            lw_seq, M, base_weight=bw_seq,
                            chunk=self.gdn2_chunk_size,
                            ckpt_chunks=self.training and self.gdn2_ckpt_chunks)
                    elif self.gdn2_chunked:
                        # Dense chunked-WY (GEMM) path -- exact reassociation
                        # of the loop below, full d_mem x d_mem state, ~26x
                        # faster. Default since 2026-07-15 (verified by
                        # temp/test_gdn2_chunked_dense.py, ~1e-6 fwd+bwd).
                        from delta_chunked import gdn2_chunked_dense
                        if self.deltaproduct_nh > 1:
                            from deltaproduct import deltaproduct_gdn2_readout
                            r_in, M = deltaproduct_gdn2_readout(
                                self.dp, gdn2_chunked_dense, spk_seq, prev_for_k,
                                spk_for_qv, v_in, k_in, b_seq, w_seq, alpha_seq, q_in, M,
                                chunk=self.gdn2_chunk_size,
                                ckpt_chunks=self.training and self.gdn2_ckpt_chunks)
                        else:
                            r_in, M = gdn2_chunked_dense(v_in, k_in, q_in, b_seq, w_seq, alpha_seq, M,
                                                         chunk=self.gdn2_chunk_size,
                                                         ckpt_chunks=self.training and self.gdn2_ckpt_chunks)
                    else:
                        r_chunks = []
                        for s in range(0, T, _GDN2_OSDN_CHUNK):
                            e = min(s + _GDN2_OSDN_CHUNK, T)
                            args = (M, k_in[:, s:e], v_in[:, s:e], q_in[:, s:e],
                                    b_seq[:, s:e], w_seq[:, s:e], alpha_seq[:, s:e])
                            if self.training:
                                M, r_ch = checkpoint(_gdn2_chunk_fwd, *args,
                                                     use_reentrant=False)
                            else:
                                M, r_ch = _gdn2_chunk_fwd(*args)
                            r_chunks.append(r_ch)
                        r_in = torch.cat(r_chunks, dim=1)
                elif self.osdn_enabled:
                    # OSDN (arXiv:2605.13473): online diagonal preconditioner on the
                    # delta write. d_t adapted online (no learned params), reset per
                    # window, but carried across chunks within the window. Composes
                    # with decay_gate. Same grad-checkpointing pattern as GDN-2.
                    beta_seq = self.beta_floor + (1 - self.beta_floor) * torch.sigmoid(self.W_beta(spk_seq))  # (B,T,1)
                    self._last_beta_stats = _beta_stats(beta_seq, floor=self.beta_floor)
                    if self.decay_gate:
                        alpha_seq_full = torch.sigmoid(self.W_alpha(spk_seq)).clamp(0.5, 0.9999)  # (B,T,1)
                        self._last_alpha_stats = _alpha_stats(alpha_seq_full)
                    else:
                        # Scalar lam broadcast to (B,T,1) so the chunk helper
                        # has one signature; cheap and keeps the loop simple.
                        alpha_seq_full = torch.full((B, T, 1), self.lam,
                                                    device=x_seq.device, dtype=x_seq.dtype)
                    d = torch.ones(B, self.d_mem, device=x_seq.device, dtype=x_seq.dtype)
                    r_chunks = []
                    for s in range(0, T, _GDN2_OSDN_CHUNK):
                        e = min(s + _GDN2_OSDN_CHUNK, T)
                        args = (M, d, k_in[:, s:e], v_in[:, s:e], q_in[:, s:e],
                                beta_seq[:, s:e], alpha_seq_full[:, s:e],
                                self.osdn_eta, self.osdn_dmin, self.osdn_dmax)
                        if self.training:
                            M, d, r_ch = checkpoint(_osdn_chunk_fwd, *args,
                                                    use_reentrant=False)
                        else:
                            M, d, r_ch = _osdn_chunk_fwd(*args)
                        r_chunks.append(r_ch)
                    r_in = torch.cat(r_chunks, dim=1)
                elif self.sdm_enabled:
                    # SDM (arXiv:2607.07386): sparse N-slot PKM-addressed memory
                    # replaces the whole dense read/write, including its own
                    # addressing -- k_in/q_in (already computed above) are unused
                    # here, only v_in (the value to write) and spk_seq (for PKM
                    # addressing, matching docs/sdm_design.md's x_t=spk mapping)
                    # are consumed. β/α stay owned by this block exactly as the
                    # plain delta_rule branch computes them.
                    #
                    # NOTE: self.sdm.M_0 (SparseSDMMemory's own learned initial
                    # state) is inert on this path -- M_init=M below is always
                    # the LM-level state allocator's zero-init tensor (or a
                    # carried M from a previous window), matching the same
                    # convention every other write rule in this codebase
                    # follows (no block owns a learned initial memory state;
                    # the LM always supplies M externally). M_0 only activates
                    # when SparseSDMMemory is driven standalone with
                    # M_init=None (see temp/test_sdm.py), which is how phase 2
                    # verified it -- expected, not a bug.
                    beta_seq = self.beta_floor + (1 - self.beta_floor) * torch.sigmoid(self.W_beta(spk_seq))  # (B,T,1)
                    self._last_beta_stats = _beta_stats(beta_seq, floor=self.beta_floor)
                    if self.decay_gate:
                        alpha_seq = torch.sigmoid(self.W_alpha(spk_seq)).clamp(0.5, 0.9999)  # (B,T,1)
                        self._last_alpha_stats = _alpha_stats(alpha_seq)
                    else:
                        alpha_seq = torch.full((B, T, 1), self.lam,
                                               device=x_seq.device, dtype=x_seq.dtype)
                    r_in, M = self.sdm(spk_seq, v_in, beta_seq, alpha_seq, M_init=M)
                elif self.vector_beta:
                    # Per-channel β (RWKV-7 style). The chunked WY kernel assumes
                    # scalar β; use a sequential memory loop instead (LIF stays
                    # FPT-parallel, only the memory update is per-timestep).
                    # Slower (≈ titans cost) but correct; use small T/batch.
                    if self.n_heads > 1:
                        raise NotImplementedError("vector_beta + n_heads>1 not supported")
                    beta_seq = self.beta_floor + (1 - self.beta_floor) * torch.sigmoid(self.W_beta(spk_seq))  # (B,T,D)
                    self._last_beta_stats = _beta_stats(beta_seq, floor=self.beta_floor)
                    alpha_seq = (torch.sigmoid(self.W_alpha(spk_seq)).clamp(0.5, 0.9999)
                                 if self.decay_gate else None)
                    if alpha_seq is not None:
                        self._last_alpha_stats = _alpha_stats(alpha_seq)
                    r_list = []
                    for t in range(T):
                        kn = k_in[:, t] / (k_in[:, t].norm(dim=-1, keepdim=True) + 1e-6)
                        a = alpha_seq[:, t].unsqueeze(2) if alpha_seq is not None else self.lam
                        prevM = a * M
                        Mk = torch.bmm(prevM, kn.unsqueeze(2)).squeeze(2)
                        u = beta_seq[:, t] * (v_in[:, t] - Mk)            # (B,D)
                        M = prevM + torch.bmm(u.unsqueeze(2), kn.unsqueeze(1))
                        r_list.append(torch.bmm(M, q_in[:, t].unsqueeze(2)).squeeze(2))
                    r_in = torch.stack(r_list, dim=1)
                else:
                    beta_seq = self.beta_floor + (1 - self.beta_floor) * torch.sigmoid(self.W_beta(spk_seq))  # (B,T,1)
                    self._last_beta_stats = _beta_stats(beta_seq, floor=self.beta_floor)
                    if self.n_heads > 1:
                        beta_in = beta_seq.unsqueeze(2).expand(B, T, self.n_heads, 1).reshape(B * self.n_heads, T, 1)
                    else:
                        beta_in = beta_seq
                    if self.decay_gate:
                        alpha_seq = torch.sigmoid(self.W_alpha(spk_seq)).clamp(0.5, 0.9999)
                        self._last_alpha_stats = _alpha_stats(alpha_seq)
                        if self.n_heads > 1:
                            alpha_in = alpha_seq.unsqueeze(2).expand(B, T, self.n_heads, 1).reshape(B * self.n_heads, T, 1)
                        else:
                            alpha_in = alpha_seq
                        if self.use_fla_kernel:
                            from delta_chunked import delta_fla_gated
                            r_in, M = delta_fla_gated(v_in, k_in, q_in, beta_in, alpha_in, M)
                        else:
                            from delta_chunked import delta_chunked_gated
                            r_in, M = delta_chunked_gated(v_in, k_in, q_in, beta_in,
                                                           alpha_in, M, chunk=64,
                                                           compile_inv=self.compile_inv)
                    else:
                        from delta_chunked import delta_chunked
                        r_in, M = delta_chunked(v_in, k_in, q_in, beta_in, M,
                                                 self.lam, chunk=64)
            elif self.learnable_decay:                       # ST Phase 4
                from hebbian_chunked import hebbian_gated_chunked
                alpha = torch.sigmoid(self.decay_raw).clamp(0.5, 0.9999)
                r_in, M = hebbian_gated_chunked(v_in, k_in, q_in, M,
                                                 alpha, self.eta, chunk=64)
            else:
                from hebbian_chunked import hebbian_chunked
                # chunk=256 (vs 64): ~4x fewer Python-launched kernels per layer,
                # which matters a lot uncompiled at long block_size (launch-bound).
                r_in, M = hebbian_chunked(v_in, k_in, q_in, M,
                                           self.lam, self.eta, chunk=256)
            # Multi-head: collapse heads back into d_mem dim.
            if self.n_heads > 1:
                H, Dh = self.n_heads, self.d_head
                r_seq = r_in.reshape(B, H, T, Dh).permute(0, 2, 1, 3).reshape(B, T, self.d_mem)
            else:
                r_seq = r_in
            if self.local_attn:
                # BASED-style sliding-window exact attention: scores = Q K^T /√D,
                # masked to (b<=t and t-b<W), softmax, then attn@V. Gated additive.
                D_ = q_seq.size(-1)
                scale = D_ ** -0.5
                scores = torch.bmm(q_seq, k_seq.transpose(1, 2)) * scale  # (B,T,T)
                idx = torch.arange(T, device=q_seq.device)
                diff = idx.unsqueeze(1) - idx.unsqueeze(0)
                m = (diff >= 0) & (diff < self.local_window)
                scores = scores.masked_fill(~m.unsqueeze(0), float('-inf'))
                attn = torch.softmax(scores, dim=-1)
                r_local = torch.bmm(attn, v_seq)
                r_seq = r_seq + torch.tanh(self.local_gate) * r_local
            out_seq = self.norm(r_seq + ff_seq)
        else:
            out_seq = self.norm(ff_seq)
        final_spk = spk_seq[:, -1, :]
        return out_seq, mem_final, M, spk_seq, final_spk


class SpikingHebbianLM(nn.Module):
    def __init__(self, vocab, d=128, n_neurons=256, d_mem=64,
                 beta=0.9, lam=0.98, eta=1.0, recurrent=False, rec_density=0.05,
                 compile_safe=False, tie_weights=False, n_layers=1,
                 use_fpt=False, fpt_K=10, learnable_decay=False, write_gate=False,
                 delta_rule=False, beta_floor=0.0, decay_gate=False,
                 titans=False, local_attn=False, local_window=64,
                 n_heads=1, pre_conv=False, pre_conv_kernel=4,
                 vector_beta=False, mtp_enabled=False, mtp_offset=2,
                 use_rmsnorm=False, embed_scale=False, mtp_depth=1,
                 use_swiglu=False, depth_scaled_init=False, final_norm=False,
                 markov_enabled=False, markov_rank=64,
                 confidence_enabled=False, compile_inv=False,
                 use_fla_kernel=False, n_experts=1, route_topk=None,
                 route_noise_std=0.0, router_z_weight=0.0,
                 ponder_enabled=False, ponder_steps=3, ponder_capacity=None,
                 ponder_threshold=0.5,
                 engram_enabled=False, engram_n=3, engram_K=32768, engram_rank=64,
                 engram_v2_enabled=False, engram_v2_orders=(2, 3),
                 engram_v2_K=65536, engram_v2_heads=2, engram_v2_rank=64,
                 recurrent_depth=1, recurrent_bptt_k=0,
                 recurrent_router_enabled=False, recurrent_router_capacity=None,
                 recurrent_router_threshold=0.5,
                 recurrent_deep_supervision=False, recurrent_pass_weights=None,
                 fracconn_enabled=False, fracconn_n=2,
                 gdn2_enabled=False, osdn_enabled=False,
                 osdn_eta=0.003, osdn_dmin=0.5, osdn_dmax=2.0,
                 sdm_enabled=False, sdm_N=None, sdm_W=4, sdm_R=4,
                 gdn2_min_heads=1, gdn2_chunked=True, gdn2_ckpt_chunks=True,
                 gdn2_chunk_size=64, gdn2_gate_norm=False,
                 gdn2_loglinear=False, gdn2_loglinear_L=None,
                 deltaproduct_nh=1, gate_rank=0, lif_exact=False, lif_reset=1.0, act_ckpt=False,
                 attention_layer_positions=None,
                 attention_heads=8,
                 attention_max_seq_len=4096,
                 attention_qk_norm=True,
                 spike_rate_undiluted=False,
                 attn_single_residual=False, telemetry_eval_only=False):
        super().__init__()
        self.vocab = vocab
        # attn_single_residual (2026-09-25 fix, default off = byte-identical to
        # every existing ckpt): HybridAttentionBlock already returns
        # x + attn + ffn, and the block loop added x AGAIN -> 2x + attn + ffn
        # (verified temp/verify_double_residual.py). True skips the loop's add
        # for attention blocks. Changes the function -> fresh runs only.
        self.attn_single_residual = attn_single_residual
        self.d = d
        self.n_neurons = n_neurons
        self.d_mem = d_mem
        self.lam = lam
        self.eta = eta
        self.n_layers = n_layers
        self._spike_rate_undiluted = spike_rate_undiluted
        # Recurrent depth (Geiping-style latent looping of the block stack).
        # recurrent_depth=1 => exact original single-pass behavior (no-op).
        # recurrent_bptt_k=0 => backprop through all passes; >0 => truncate to
        # the last k passes (bounded memory at large depth, Huginn-style).
        self.recurrent_depth = recurrent_depth
        self.recurrent_bptt_k = recurrent_bptt_k
        # Stage 2: per-token adaptive depth (MoR capacity router gating which
        # tokens continue past each pass boundary) -- see RecurrentDepthRouter.
        # False (default) => byte-identical to Stage 1 (every token gets all
        # recurrent_depth passes).
        self.recurrent_router_enabled = recurrent_router_enabled
        if recurrent_router_enabled:
            assert recurrent_depth >= 2, "recurrent_router_enabled requires recurrent_depth >= 2"
            from adaptive_compute import RecurrentDepthRouter
            self.rdepth_router = RecurrentDepthRouter(
                d_mem, recurrent_depth, capacity=recurrent_router_capacity,
                threshold=recurrent_router_threshold)
        self._last_rdepth_aux = None
        self._last_rdepth_depth = None
        # Phase 1 (recurrent-depth fix, 2026-07-06): per-pass deep supervision.
        # Every successful looped-model recipe in the 2025-26 literature (Ouro,
        # LoopFormer) applies the LM loss at every pass exit, not just the
        # final one -- without it, early passes never get direct training
        # signal as decodable outputs, which likely explains the depth-2
        # screens' unusually slow convergence (still falling steeply at
        # budget end while sibling configs plateau). recurrent_pass_weights
        # is a list of length recurrent_depth-1 (one weight per NON-final
        # pass, in pass order); None -> uniform 0.3 per intermediate pass.
        # The final pass needs no extra term here -- it's already the primary
        # CE the trainer computes on the model's returned logits.
        self.recurrent_deep_supervision = recurrent_deep_supervision
        self.recurrent_pass_weights = recurrent_pass_weights
        self._last_pass_logits = None
        self.compile_safe = compile_safe
        self.tie_weights = tie_weights
        self.use_fpt = use_fpt
        self.fpt_K = fpt_K
        # act_ckpt: recompute each block's forward in backward (torch.utils.checkpoint,
        # non-reentrant) to trade (idle, launch-bound) GPU time for activation VRAM ->
        # bigger micro-batch. Single-pass training path only. Default off.
        self.act_ckpt = act_ckpt

        if n_layers > 1 and d != d_mem:
            raise ValueError(f"n_layers > 1 requires d == d_mem for residual connections "
                             f"(got d={d}, d_mem={d_mem})")
        self.n_experts = n_experts
        if n_experts > 1 and not use_fpt:
            raise ValueError("n_experts > 1 (Hebbian-MoE) only supports the FPT/chunked "
                             "path -- set use_fpt: true")

        self.embed = nn.Embedding(vocab, d)
        # Llama / DeepSeek convention: when embedding output is later multiplied
        # by sqrt(d), the embedding itself must be initialized with std=1/sqrt(d)
        # so the product lands at O(1). Without this paired init, embed_scale=True
        # blows up the loss (verified empirically — 30M Batch A diverged on this).
        if embed_scale:
            nn.init.normal_(self.embed.weight, std=1.0 / math.sqrt(d))
        # Batch B: depth-scaled init factor for output projections.
        # Each block's residual-adjacent Linear has its weight multiplied by
        # 1/sqrt(2*n_layers) — bounds residual-stream variance under depth.
        depth_scale = 1.0 / math.sqrt(2 * n_layers) if depth_scaled_init else 1.0
        # Hybrid attention: replace specified layer indices with a full-softmax
        # Transformer block (RoPE + SwiGLU). Fixes the pure-linear coherence
        # gap while keeping Hebbian layers as the differentiator. Follows the
        # SpikingBrain-2025 / MiniMax-01 / Jamba layout of interleaving full
        # attention among linear layers. See docs/model_quality_redesign.md P2.
        attn_positions = set(attention_layer_positions or [])
        if attn_positions and (min(attn_positions) < 0 or max(attn_positions) >= n_layers):
            raise ValueError(f"attention_layer_positions {attn_positions} out of "
                             f"range for n_layers={n_layers}")
        if 0 in attn_positions and d != d_mem:
            # HybridAttentionBlock requires d_in == d_mem; layer 0 has d_in = d.
            # When d != d_mem the layer 0 adapter shape mismatch is fatal. When
            # d == d_mem it's fine, and the matched-transformer baseline needs it.
            raise ValueError("attention_layer_positions cannot include 0 when "
                             f"d ({d}) != d_mem ({d_mem}); layer 0 adapts d -> d_mem")

        blocks = []
        for i in range(n_layers):
            d_in = d if i == 0 else d_mem
            if i in attn_positions:
                from hybrid_attn import HybridAttentionBlock
                blocks.append(HybridAttentionBlock(
                    d_in=d_in, d_mem=d_mem, n_neurons=n_neurons,
                    n_heads=attention_heads, use_rmsnorm=use_rmsnorm,
                    use_swiglu=use_swiglu, depth_scale=depth_scale,
                    max_seq_len=attention_max_seq_len,
                    use_qk_norm=attention_qk_norm))
            elif n_experts > 1:
                from moe_hebbian import MoEHebbianBlock
                blocks.append(MoEHebbianBlock(
                    d_in, n_neurons, d_mem, n_experts, beta,
                    beta_floor=beta_floor, use_rmsnorm=use_rmsnorm,
                    use_swiglu=use_swiglu, depth_scale=depth_scale,
                    compile_inv=compile_inv, use_fla_kernel=use_fla_kernel,
                    route_topk=route_topk, route_noise_std=route_noise_std,
                    router_z_weight=router_z_weight))
            else:
                blocks.append(SpikingHebbianBlock(
                    d_in, n_neurons, d_mem, beta, lam, eta,
                    compile_safe, recurrent, rec_density,
                    learnable_decay=learnable_decay, write_gate=write_gate,
                    delta_rule=delta_rule, beta_floor=beta_floor,
                    decay_gate=decay_gate, titans=titans,
                    local_attn=local_attn, local_window=local_window,
                    n_heads=n_heads,
                    pre_conv=pre_conv, pre_conv_kernel=pre_conv_kernel,
                    vector_beta=vector_beta, use_rmsnorm=use_rmsnorm,
                    use_swiglu=use_swiglu, depth_scale=depth_scale,
                    compile_inv=compile_inv, use_fla_kernel=use_fla_kernel,
                    gdn2_enabled=gdn2_enabled, osdn_enabled=osdn_enabled,
                    osdn_eta=osdn_eta, osdn_dmin=osdn_dmin, osdn_dmax=osdn_dmax,
                    sdm_enabled=sdm_enabled, sdm_N=sdm_N, sdm_W=sdm_W, sdm_R=sdm_R,
                    gdn2_min_heads=gdn2_min_heads, gdn2_chunked=gdn2_chunked,
                    gdn2_ckpt_chunks=gdn2_ckpt_chunks, gdn2_chunk_size=gdn2_chunk_size,
                    gdn2_gate_norm=gdn2_gate_norm,
                    gdn2_loglinear=gdn2_loglinear,
                    gdn2_loglinear_L=gdn2_loglinear_L,
                    deltaproduct_nh=deltaproduct_nh,
                    gate_rank=gate_rank, lif_exact=lif_exact, lif_reset=lif_reset))
        self.titans = titans
        self.n_heads = n_heads
        self.d_head = d_mem // n_heads
        self.blocks = nn.ModuleList(blocks)
        for b in self.blocks:
            if isinstance(b, SpikingHebbianBlock):
                b.telemetry_eval_only = telemetry_eval_only
        # Frac-Connections (arXiv:2503.14125, see docs/hyperconnections_design.md):
        # per-layer learnable generalization of the residual add, identity-init
        # so it's a no-op until trained. One instance per layer (state is
        # per-layer, not per recurrent-depth iteration -- composes cleanly with
        # the router loop, which re-enters the same layer module every pass).
        self.fracconn_enabled = fracconn_enabled
        if fracconn_enabled:
            from fracconn import FracConnection
            self.fracconns = nn.ModuleList(
                [FracConnection(d_mem, n_frac=fracconn_n) for _ in range(n_layers)])
        self.head = nn.Linear(d_mem, vocab)
        # Final pre-head norm (2026-08-26). Default off => spiking path
        # unchanged. Required for attention-heavy stacks (matched-
        # transformer baseline): 14 pre-norm attention layers grow the
        # residual stream unbounded, and self.head consumes it raw.
        self.out_norm = RMSNorm(d_mem) if final_norm else None
        if tie_weights:
            if d != d_mem:
                raise ValueError(f"tie_weights requires d == d_mem (got d={d}, d_mem={d_mem})")
            self.head.weight = self.embed.weight

        # Multi-Token Prediction (MTP) auxiliary head — DeepSeek-V3 style.
        # When enabled, the same hidden state is projected through a small
        # transform and re-fed to self.head to predict tokens at position
        # t+mtp_offset (default t+2). Adds ~d_mem*d_mem params (~4M at d_mem=2048).
        # Loss is added in the trainer; the projected logits are stashed on the
        # module after each forward for the trainer to consume.
        # Batch A: embedding-scale by sqrt(d) (Llama / DeepSeek convention).
        # Small stability win; gated by config to keep ckpt compat.
        self.embed_scale = embed_scale
        self._embed_scale_factor = math.sqrt(d) if embed_scale else 1.0

        self.mtp_enabled = mtp_enabled
        self.mtp_offset = mtp_offset
        self.mtp_depth = mtp_depth
        if mtp_enabled:
            self.mtp_norm = _make_norm(d_mem, use_rmsnorm)
            if mtp_depth >= 2:
                # DeepSeek-V3 style: deeper MTP head with hidden expansion.
                # Adds ~2x params vs single Linear, cleaner t+2 signal.
                self.mtp_proj = nn.Sequential(
                    nn.Linear(d_mem, d_mem),
                    nn.GELU(),
                    nn.Linear(d_mem, d_mem),
                )
            else:
                self.mtp_proj = nn.Linear(d_mem, d_mem)
        self._last_mtp_logits = None

        # Batch C: Markov head (DSpark-inspired last-token bias correction).
        # Adds a low-rank correction to logits based on the immediately-previous
        # token. Complements Hebbian state by giving the model an explicit,
        # non-decayed "what did I just emit" signal. Params: 2 * vocab * rank.
        self.markov_enabled = markov_enabled
        if markov_enabled:
            self.markov_embed  = nn.Embedding(vocab, markov_rank)
            self.markov_out    = nn.Linear(markov_rank, vocab, bias=False)
            # Small init so it doesn't dominate early
            nn.init.normal_(self.markov_embed.weight, std=0.02)
            nn.init.normal_(self.markov_out.weight, std=0.02)

        # Batch C: Confidence head. Per-position scalar predicting whether the
        # argmax matches the target. Used at inference to gate self-refine,
        # brain-mod switching, or adaptive computation. During training we add
        # a small BCE loss vs the correctness signal.
        self.confidence_enabled = confidence_enabled
        if confidence_enabled:
            self.confidence_head = nn.Linear(d_mem, 1)
        self._last_confidence = None
        self._last_moe_aux_loss = None
        self._last_expert_util = None
        self._last_route_peak = None

        # Adaptive computation: Mixture-of-Recursions expert-choice refinement of
        # the final hidden state. Hard top-k capacity per depth -> collapse-proof
        # (replaces the earlier PonderNet halting head, which collapsed). See
        # src/adaptive_compute.py.
        self.ponder_enabled = ponder_enabled
        if ponder_enabled:
            from adaptive_compute import MoRRefine
            self.ponder = MoRRefine(d_mem, ponder_steps=ponder_steps,
                                    capacity=ponder_capacity,
                                    threshold=ponder_threshold,
                                    use_rmsnorm=use_rmsnorm)
        self._last_ponder_aux = None
        self._last_expected_steps = None

        # Engram: n-gram hash -> low-rank bucket -> vocab bias (DeepSeek-style
        # static factual store). Sibling of the Markov head but higher-order
        # context (last N tokens, not just current). Zero-init buckets so it
        # contributes nothing until it learns. See src/engram.py.
        self.engram_enabled = engram_enabled
        if engram_enabled:
            from engram import EngramLookup
            self.engram = EngramLookup(vocab, n=engram_n, K=engram_K, rank=engram_rank)

        # Engram v2: residual-stream conditional memory (DeepSeek V4 alignment).
        # Adds a retrieved d-dim vector into the INPUT residual (before the block
        # stack), so knowledge participates in downstream reasoning -- not a
        # vocab logit bias like v1. Multi-head hashing + multi-granular n-grams
        # for collision robustness. Zero-init projection: exact no-op at start,
        # safe to enable on a fresh run or a fine-tune. See src/engram_v2.py.
        # v1 and v2 are independent and can coexist; use v2 alone for the
        # redesign A/B (see docs/to-do.html idea C).
        self.engram_v2_enabled = engram_v2_enabled
        if engram_v2_enabled:
            from engram_v2 import EngramV2
            self.engram_v2 = EngramV2(
                d=d, vocab=vocab,
                orders=tuple(engram_v2_orders),
                K=engram_v2_K, heads=engram_v2_heads, rank=engram_v2_rank)

    def _attn_owns_residual(self, block):
        """True when the loop must NOT add the residual: attn_single_residual is
        on and the block is a HybridAttentionBlock (it adds its own)."""
        return self.attn_single_residual and hasattr(block, "qkv")

    def _fpt_stack_pass(self, x_in, ablate_memory, return_stats):
        """One pass of the full block stack over the sequence, with FRESH memory.

        This is the recurrent-depth "core": looping it lets the model refine its
        latent representation through the ACTUAL Hebbian blocks (sharing their
        weights and their memory machinery), rather than through a separate
        bolted-on module. Because each pass is an ordinary block-stack forward
        with the real gated-delta memory, ablating memory degrades it the same
        graceful, bounded way the base model degrades -- unlike the old
        final-hidden-state ponder bolt-on, whose ablation error grew unbounded.

        Fresh memory per pass (not carried across passes): the memory is rebuilt
        from the current refined representation each pass, which shares the
        memory *machinery* without double-writing a running state across passes.

        Returns (x_seq, spikes, aux, mems, Ms, final_spks) -- the per-layer
        memory states from THIS pass, so the caller can expose the LAST pass's
        states as the model's final state (return_final_state / brain
        snapshots). The last pass's memory -- built from the most refined
        representation -- IS the model's memory after reading the sequence;
        discarding it (the original behavior) made recurrent-depth brain
        snapshots silently all-zero.
        """
        B = x_in.shape[0]
        device = x_in.device
        L = self.n_layers
        use_residual = (self.d == self.d_mem)
        mems = [torch.zeros(B, self.n_neurons, device=device) for _ in range(L)]
        Ms = [torch.zeros(B * blk.n_state_units, blk.d_state, blk.d_state,
                          device=device) for blk in self.blocks]
        prev_spks = [torch.zeros(B, self.n_neurons, device=device) for _ in range(L)]
        spikes, aux, final_spks = [], [], []
        x_seq = x_in
        for l, block in enumerate(self.blocks):
            out_seq, mems[l], Ms[l], spk_seq, final_spk = block.forward_sequence(
                x_seq, mems[l], Ms[l], prev_spks[l],
                ablate_memory=ablate_memory, fpt_K=self.fpt_K)
            if L > 1 and (l > 0 or use_residual) and not self._attn_owns_residual(block):
                out_seq = self.fracconns[l](out_seq, x_seq) if self.fracconn_enabled else out_seq + x_seq
            x_seq = out_seq
            if return_stats:
                spikes.append(spk_seq)
            if getattr(block, "_last_aux_loss", None) is not None:
                aux.append(block._last_aux_loss)
            final_spks.append(final_spk)
        return x_seq, spikes, aux, mems, Ms, final_spks

    def forward(self, idx, ablate_memory=False, return_stats=False,
                initial_state=None, return_final_state=False,
                ablate_engram=False, ablate_engram_v2=False,
                return_mtp=False, kv_cache=None):
        B, T = idx.shape
        device = idx.device
        # Incremental-decode KV cache (inference only). A list of length n_layers;
        # attention layers read/UPDATE-IN-PLACE their (k,v) entry, spiking layers
        # ignore it and carry state via initial_state as usual. When None (all
        # training + every existing caller) the code below is byte-identical.
        # Generation-only, single-pass FPT path only.
        if kv_cache is not None:
            if return_stats:
                raise ValueError("kv_cache (incremental decode) does not support return_stats")
            if not self.use_fpt or self.recurrent_depth > 1:
                raise ValueError("kv_cache path requires use_fpt and recurrent_depth == 1")
        base = self.embed(idx) * self._embed_scale_factor      # (B, T, d)
        # Engram v2: residual-stream injection (DeepSeek-style). Add BEFORE the
        # block stack so the retrieved knowledge participates in all downstream
        # attention/mixing/Hebbian dynamics. Zero-init proj -> exact no-op at
        # start; only starts contributing as the model trains it up.
        if self.engram_v2_enabled and not ablate_engram_v2:
            base = base + self.engram_v2(idx)
        L = self.n_layers

        # An initial_state (any caller -- stateful training, recall_eval's
        # split-replay, etc.) forces the single-pass path below, which SILENTLY
        # drops the recurrent loop (and its router). Fail loudly instead of
        # letting a depth>1 model get evaluated/used at depth 1 unnoticed.
        if self.recurrent_depth > 1 and initial_state is not None:
            raise ValueError("recurrent_depth > 1 is silently bypassed when "
                             "initial_state is passed (single-pass path only); "
                             "recurrent-depth models need a multi-pass-with-state "
                             "path, not yet implemented")

        # Recurrent-depth branch condition, hoisted so the state init below can
        # skip its (otherwise dead) zero-allocation: _fpt_stack_pass builds its
        # own fresh states each pass and the LAST pass's states are adopted as
        # the model's final state after the loop.
        use_rdepth = (self.use_fpt and self.recurrent_depth > 1
                      and initial_state is None)

        if initial_state is None:
            if use_rdepth:
                # Populated from the final stack pass in the recurrent branch --
                # pre-allocating here was ~2*B*d_mem^2 floats of zero-init churn
                # per forward that the branch never read.
                mems = Ms = prev_spks = None
                Ss = [None] * L
            else:
                mems = [torch.zeros(B, self.n_neurons, device=device) for _ in range(L)]
                # Each block owns its own state shape (n_heads>1 collapses split
                # heads into batch; MoE collapses full-size experts into batch).
                Ms = [torch.zeros(B * blk.n_state_units, blk.d_state, blk.d_state,
                                  device=device) for blk in self.blocks]
                prev_spks = [torch.zeros(B, self.n_neurons, device=device) for _ in range(L)]
                Ss = [torch.zeros(B * self.n_heads, self.d_head, self.d_head, device=device)
                      for _ in range(L)] if self.titans else [None] * L
        else:
            def _bcast(t, shape):
                # 2026-08-09 fix (audit finding D6, escalated from "wasteful" to
                # a real OOM blocker): the caller now passes each BLOCK's own
                # state shape (matching the initial_state-is-None branch above)
                # instead of a hardcoded (B, d_mem, d_mem) for every layer. That
                # hardcoding forced HybridAttentionBlock's tiny (B,1,1) dummy M
                # through .expand(...).contiguous() into a full materialized
                # (B, d_mem, d_mem) tensor -- 1.27 GB from nothing at B=256,
                # d_mem=1152 -- and OOM'd recall_eval's split-replay pass 2
                # (temp/diag_recall_oom.py pinned it to this exact allocation).
                # Also skip the copy entirely when t already has the right
                # shape and is contiguous (the common split-replay case, where
                # pass 1's own carried state already matches B): .expand() on
                # already-equal dims plus .contiguous() still forces a full
                # redundant copy of every spiking block's REAL M state, which
                # was the other ~8 GB of the same OOM.
                if t.dim() == len(shape) - 1: t = t.unsqueeze(0)
                t = t.to(device)
                if tuple(t.shape) == tuple(shape) and t.is_contiguous():
                    return t
                return t.expand(*shape).contiguous()
            layers = initial_state.get("layers", [initial_state])
            mems = [_bcast(layers[i]["mem"], (B, self.n_neurons)) for i in range(L)]
            Ms = [_bcast(layers[i]["M"], (B * blk.n_state_units, blk.d_state, blk.d_state))
                  for i, blk in enumerate(self.blocks)]
            prev_spks = [_bcast(layers[i]["prev_spk"], (B, self.n_neurons)) for i in range(L)]
            if self.titans:
                Ss = [_bcast(layers[i].get("S", torch.zeros(self.d_mem, self.d_mem)),
                             (B, self.d_mem, self.d_mem)) for i in range(L)]
            else:
                Ss = [None] * L

        use_residual = (self.d == self.d_mem)

        if self.use_fpt:
            spike_accum = []
            moe_aux_losses = []
            if use_rdepth:
                # Recurrent depth: loop the block stack R times, sharing weights,
                # with input injection (the embedded input `e` is re-added each
                # pass -- anchors the recursion to the input and aids gradient
                # flow, per Geiping et al. / Huginn). Latent state starts at 0;
                # pass i computes s_{i+1} = Stack(s_i + e) with fresh memory.
                e = base
                x_seq = torch.zeros_like(e)
                R = self.recurrent_depth
                k = self.recurrent_bptt_k if self.recurrent_bptt_k > 0 else R
                if self.recurrent_router_enabled:
                    active = torch.ones(B, T, dtype=torch.bool, device=device)
                    depth_bt = None
                    self.rdepth_router.reset()
                if self.recurrent_deep_supervision:
                    self._last_pass_logits = []
                for pass_i in range(R):
                    # Truncated BPTT: detach the state before the last k passes
                    # so gradient memory stays bounded at large depth.
                    if pass_i < R - k:
                        x_seq = x_seq.detach()
                    last = (pass_i == R - 1)
                    new_x_seq, spikes, aux, p_mems, p_Ms, p_spks = self._fpt_stack_pass(
                        x_seq + e, ablate_memory, return_stats and last)
                    if self.recurrent_router_enabled:
                        if pass_i == 0:
                            # Pass 0 is mandatory -- every token gets it (depth
                            # >= 1 always). Gating starts at the first boundary.
                            x_seq = new_x_seq
                            depth_bt = torch.ones(B, T, device=device)
                        else:
                            x_seq, active = self.rdepth_router.gate(
                                pass_i - 1, x_seq, new_x_seq, active)
                            depth_bt = depth_bt + active.float()
                    else:
                        x_seq = new_x_seq
                    if self.recurrent_deep_supervision and not last and self.training:
                        # Project THIS pass's actual representation (post router
                        # merge, so frozen tokens are graded on their real exit
                        # state, not the discarded recompute) -- trains each
                        # pass to be independently decodable, not just a latent
                        # stepping-stone the final pass compensates for.
                        # Gated on self.training: eval doesn't need this loss
                        # term, and retaining (B,T,V) logits per non-final pass
                        # was extra vocab-size memory/compute on every eval call.
                        self._last_pass_logits.append(self.head(x_seq))
                    if last:
                        spike_accum = spikes
                        moe_aux_losses = aux
                        # Adopt the final pass's memory as the model's state
                        # (return_final_state / brain snapshots) -- previously
                        # discarded, leaving all-zero states.
                        mems, Ms, prev_spks = p_mems, p_Ms, p_spks
                if self.recurrent_router_enabled:
                    self.rdepth_router.finalize(depth_bt)
                    self._last_rdepth_aux = self.rdepth_router._last_aux
                    self._last_rdepth_depth = self.rdepth_router._last_expected_depth
            else:
                # Single-pass (original behavior; also the path for stateful
                # inference with an initial_state). Unchanged.
                x_seq = base                                   # (B, T, d)
                for l, block in enumerate(self.blocks):
                    if kv_cache is not None and hasattr(block, "qkv"):
                        # Incremental attention: use the KV cache instead of a
                        # full re-forward. step_kv is proven equivalent to this
                        # block's forward_sequence (temp/test_hybrid_kv_parity.py),
                        # so the residual logic below is unchanged. Attention has
                        # no recurrent state, so mem/M/prev_spk pass through as
                        # forward_sequence does; the cache is updated IN PLACE so
                        # the caller carries it across steps with no return change.
                        out_seq, kv_cache[l] = block.step_kv(x_seq, kv_cache[l])
                        spk_seq = None
                    elif self.act_ckpt and self.training and torch.is_grad_enabled():
                        out_seq, mems[l], Ms[l], spk_seq, final_spk = checkpoint(
                            block.forward_sequence, x_seq, mems[l], Ms[l], prev_spks[l],
                            ablate_memory=ablate_memory, fpt_K=self.fpt_K,
                            use_reentrant=False)
                        prev_spks[l] = final_spk
                    else:
                        out_seq, mems[l], Ms[l], spk_seq, final_spk = block.forward_sequence(
                            x_seq, mems[l], Ms[l], prev_spks[l],
                            ablate_memory=ablate_memory, fpt_K=self.fpt_K)
                        prev_spks[l] = final_spk
                    if L > 1 and (l > 0 or use_residual) and not self._attn_owns_residual(block):
                        out_seq = self.fracconns[l](out_seq, x_seq) if self.fracconn_enabled else out_seq + x_seq
                    x_seq = out_seq
                    if return_stats:
                        spike_accum.append(spk_seq)
                    if getattr(block, "_last_aux_loss", None) is not None:
                        moe_aux_losses.append(block._last_aux_loss)
            self._last_moe_aux_loss = (torch.stack(moe_aux_losses).mean()
                                       if moe_aux_losses else None)
            # Diagnostic snapshot (last batch only) for judging real vs
            # trivial-uniform routing -- not used in any loss.
            self._last_expert_util = [b._last_route_util for b in self.blocks
                                      if getattr(b, "_last_route_util", None) is not None] or None
            self._last_route_peak = [b._last_route_peak for b in self.blocks
                                     if getattr(b, "_last_route_peak", None) is not None] or None
            # Adaptive computation: refine the final hidden state with a soft,
            # per-token-variable number of passes. Everything downstream (head,
            # MTP, confidence) then reads the refined representation.
            if self.ponder_enabled:
                x_seq = self.ponder(x_seq)
                self._last_ponder_aux = self.ponder._last_ponder_aux
                self._last_expected_steps = self.ponder._last_expected_steps
            else:
                self._last_ponder_aux = None
            if self.out_norm is not None:
                x_seq = self.out_norm(x_seq)
            logits = self.head(x_seq)                          # (B, T, vocab)
            if self.mtp_enabled:
                self._last_mtp_logits = self.head(self.mtp_proj(self.mtp_norm(x_seq)))
            else:
                self._last_mtp_logits = None
            # Batch C: Markov head — add correction based on the input token
            # at each position (which is the "previous token" from the
            # next-token prediction's perspective).
            if self.markov_enabled:
                markov_correction = self.markov_out(self.markov_embed(idx))  # (B, T, V)
                logits = logits + markov_correction
            if self.engram_enabled and not ablate_engram:
                logits = logits + self.engram(idx)          # (B, T, V) n-gram bias
            # Batch C: Confidence head. Stash for the trainer.
            if self.confidence_enabled:
                self._last_confidence = self.confidence_head(x_seq).squeeze(-1)  # (B, T)
            else:
                self._last_confidence = None
            all_spikes = spike_accum
        else:
            logits_list = []
            all_spikes = []
            for t in range(T):
                x = base[:, t, :]
                for l, block in enumerate(self.blocks):
                    if block.titans:
                        out, mems[l], Ms[l], Ss[l], spk = block.step_titans(
                            x, mems[l], Ms[l], Ss[l], prev_spks[l], ablate_memory)
                    else:
                        out, mems[l], Ms[l], spk = block.step(
                            x, mems[l], Ms[l], prev_spks[l], ablate_memory)
                    if L > 1 and (l > 0 or use_residual):
                        out = self.fracconns[l](out, x) if self.fracconn_enabled else out + x
                    prev_spks[l] = spk
                    x = out
                logits_list.append(self.head(self.out_norm(x) if self.out_norm is not None else x))
                if return_stats:
                    all_spikes.extend(prev_spks)
            logits = torch.stack(logits_list, dim=1)
            if self.mtp_enabled:
                # Per-token path: reconstruct x_seq-equivalent from logits_list
                # is awkward; for training we expect FPT path. Disable MTP here
                # by leaving _last_mtp_logits as None (trainer skips on None).
                self._last_mtp_logits = None
            else:
                self._last_mtp_logits = None

        spike_rate = None
        if return_stats:
            if self.use_fpt:
                # Dilution fix (2026-08-08, audit A/B2/§4.2). Attention layers
                # push (B,T,N) zero stubs into all_spikes, so the flat mean is
                # (n_spiking/n_total) * true spiking rate -- reported number
                # under-states real activity by ~14% at L=8 attn=1, ~11% at L=9
                # attn=1. The training sparsity penalty consumes this number,
                # so it was penalizing a diluted target. Behind
                # spike_rate_undiluted (default False): flag OFF is byte-
                # identical to every ckpt through P4; flag ON averages over
                # spiking-block spikes only (matches the [spike] "spiking_only"
                # telemetry line the trainer prints).
                if self._spike_rate_undiluted:
                    per_block = [s.mean() for s in all_spikes]
                    active = [s for s, b in zip(per_block, self.blocks)
                              if not hasattr(b, "qkv")]
                    spike_rate = torch.stack(active).mean() if active else torch.stack(per_block).mean()
                else:
                    spike_rate = torch.stack([s.mean() for s in all_spikes]).mean()
            else:
                spike_rate = torch.stack(all_spikes).mean()
        # return_mtp=True: pass MTP logits back as an extra tuple element so
        # trainer/callers can consume them WITHOUT reading `raw_model._last_mtp_logits`
        # (attribute writes inside a compiled forward do not reliably propagate
        # to the outer module reference under torch.compile / CUDA-graph replay
        # -- observed as MTP loss silently dropping to None under whole-model
        # compile, ~30% training signal loss). Keeps `_last_mtp_logits`
        # unchanged for legacy eager readers.
        _mtp = self._last_mtp_logits if self.mtp_enabled else None
        if return_final_state:
            state = {"layers": [
                {"M": Ms[i].detach(), "mem": mems[i].detach(), "prev_spk": prev_spks[i].detach()}
                for i in range(L)
            ]}
            # Stateful training needs both the spike rate (for the sparsity
            # penalty) and the carried state in a single forward pass.
            if return_stats:
                ret = (logits, spike_rate, state)
            else:
                ret = (logits, state)
        elif return_stats:
            ret = (logits, spike_rate)
        else:
            ret = logits
        if return_mtp:
            return (ret if isinstance(ret, tuple) else (ret,)) + (_mtp,)
        return ret

    def save_brain(self, state, path):
        """Persist a complete neuron state ('brain') to disk."""
        def _to1(t):
            if t.dim() >= 2 and t.shape[0] == 1: return t.squeeze(0)
            return t
        layers = state.get("layers", [state])
        blob = {
            "layers": [{
                "M": _to1(lay["M"]).detach().cpu(),
                "mem": _to1(lay["mem"]).detach().cpu(),
                "prev_spk": _to1(lay["prev_spk"]).detach().cpu(),
            } for lay in layers],
            "n_layers": len(layers),
            "d_mem": self.d_mem,
            "n_neurons": self.n_neurons,
            "vocab": self.vocab,
            "lam": self.lam,
            "eta": self.eta,
            "format_version": 3,
        }
        torch.save(blob, path)

    @staticmethod
    def load_brain(path):
        """Load a brain file. Returns a state dict suitable for initial_state=."""
        blob = torch.load(path, weights_only=False)
        if blob.get("format_version", 1) < 3:
            return {"layers": [{"M": blob["M"], "mem": blob["mem"], "prev_spk": blob["prev_spk"]}]}
        return {"layers": blob["layers"]}
