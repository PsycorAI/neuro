"""stream_decode.py -- 2026-10-24 (Neuro 77, Demo plan v2 W1/W2). Incremental (streaming) decoding OUTSIDE src/model.py and
src/hybrid_attn.py (the 5090 queue re-imports those on any restart, so they are never edited for demo work). Drives the
existing modules of a built model, mirroring LM.forward's single-pass loop exactly.

V4FDecoder -- all-attention models (v4f): a PREALLOCATED static KV cache (no torch.cat growth), prefill with causal SDPA,
then one token per step attending over the cache up to a length bucket (keys past the current position masked), one
CUDA graph per bucket. This is the fair, standard transformer decode the comparison gives v4f.
  dec = V4FDecoder(model, max_len=4096); logits = dec.prefill(ids); logits = dec.step(tok) ...
parity: temp/n77_stream_parity.py (decode logits vs the full forward on the same weights)."""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

SEG_SIZES = (512, 256, 128, 64, 32, 16, 8, 4)  # n79 V4Decoder.feed_static segment graphs (< 4 left: step()s, cheaper)
RING_STEP = 256                                # n79 ring-prefix lengths a segment reads while the window fills


def _rope(x, cos, sin):
    """RoPE on (B, H, t, D) with cos / sin already gathered for the t positions: (t, D)."""
    half = x.shape[-1] // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos.to(x.dtype)[None, None] + rot * sin.to(x.dtype)[None, None]


def _head(m, x):
    """LM.forward's tail (out_norm -> sc_head -> head_in -> head); asserts the unsupported extras are off."""
    if m.out_norm is not None:
        x = m.out_norm(x)
    h = m.sc_head(x) if m.sc_head is not None else x
    if m.head_in is not None:
        h = m.head_in(h)
        if getattr(m, "sc_head_mid", None) is not None:
            h = m.sc_head_mid(h)
    return m.head(h)


def _sc_forward_nostats(self, x):
    """SpikeCode.forward (src/spike_code.py) with the SAME output ops but without its two training-telemetry stats
    (last_count / last_nz: abs, float, 2 means, 2 copies per call; nothing at inference reads them) -- ~600 of a v4
    decode step's ~2,400 tiny kernels (logs/n78_step_prof.log). Calibration passes keep the original."""
    if self.calib is not None:
        return type(self).forward(self, x)
    th = self.theta.abs().to(x.dtype) + 1e-4
    u = x / th
    lo = -self.L if self.signed else 0
    s = u.clamp(lo, self.L)
    s = s + (s.round() - s).detach()
    return s * th


def strip_spike_code_stats(model):
    """n78 10-25, inference only (demo / harness): every SpikeCode of `model` runs _sc_forward_nostats (outputs bit-
    identical; forward hooks such as the demo Tracer's still fire). Returns the number of modules patched."""
    import types
    from spike_code import SpikeCode
    n = 0
    for mod in model.modules():
        if isinstance(mod, SpikeCode):
            mod.forward = types.MethodType(_sc_forward_nostats, mod); n += 1
    return n


def _check_lm(m):
    bad = [k for k in ("markov_enabled", "engram_enabled", "mtp_enabled", "ponder_enabled", "fracconn_enabled")
           if getattr(m, k, False)]
    assert not bad and m.mem_read_proj is None and m.recurrent_depth == 1 and m.use_fpt, f"unsupported LM flags {bad}"


class V4FDecoder:
    def __init__(self, model, max_len=4096, bucket=256, graphs=True):
        m = self.m = model
        _check_lm(m)
        assert not m.engram_v2_enabled, "engram_v2 needs token history: use the v4 decoder"
        for b in m.blocks:
            assert hasattr(b, "qkv"), "V4FDecoder is for all-attention models"
            assert not (b.attn_canon or b.attn_chunk or b.attn_topk or b.attn_window or len(b.sc) or b.attn_gate is not None), \
                "canon / chunk / top-k / window / spike codes / gate need the v4 decoder"
        b0 = m.blocks[0]
        self.H, self.Dh = b0.n_heads, b0.d_head
        p = next(m.parameters())
        self.dev, self.dt = p.device, p.dtype
        n = len(m.blocks)
        assert max_len <= b0.rope_cos.shape[0], "rope table shorter than max_len (raise attention_max_seq_len)"
        self.max_len, self.bucket = max_len, bucket
        # n79: one tensor per layer (was one (n, ...) tensor indexed per layer): torch.compile functionalises the
        # index_copy_ into a VIEW of a graph input by rebuilding the whole base -> a 1.64 GiB clone of the 32k cache
        # per step (OOM in the fused arm); a whole per-layer tensor is updated in place
        self.K = [torch.zeros(1, self.H, max_len, self.Dh, device=self.dev, dtype=self.dt) for _ in range(n)]
        self.V = [torch.zeros_like(k) for k in self.K]
        self.pos = 0
        self.use_graphs, self.graphs, self.pool = graphs and self.dev.type == "cuda", {}, None
        self.s_tok = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        self.s_pos = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.s_mask = {}
        # n79 10-25 (Glenn: a FUSED v4f arm, quote fused v4 vs fused v4f): compile_step = the step through torch.compile as
        # V4Decoder._body; step_backend = the SDPA backend of the step (None = PyTorch's pick: with the mask that is the
        # mem-efficient kernel, ~120 GB/s for one query = 28 ms / token of attention at 32k on the 5080; cuDNN runs the same
        # masked call at ~870 GB/s, 3.9 ms: temp/n79_attn_bench.py, logs/n79_attn_bench.log)
        # n81 10-26: cuDNN is the DEFAULT (bf16 / fp16 on CUDA) and _blocks calls it directly -- an sdpa_kernel context
        # (or enable_mem_efficient_sdp(False)) does not reach the compiled step, which ran mem-efficient on the 5090
        # (36 words/s at 32k, logs/n80_prof.log); step_backend = None restores PyTorch's pick
        self.compile_step = False
        self.step_backend = (SDPBackend.CUDNN_ATTENTION if self.dev.type == "cuda" and self.dt in (torch.bfloat16, torch.float16)
                             else None)
        self.graph_prefill, self.pgraphs = False, {}          # n79: _prefill_graph (off: the eager prefill)
        self.compile_read = False                              # n81: _prefill_graph's body through torch.compile (as v4's)

    def reset(self):
        self.pos = 0

    def _blocks(self, x, posv, kvlen, mask, causal, last_only=False, hidden=False):
        m = self.m
        use_res = (m.d == m.d_mem)
        for l, b in enumerate(m.blocks):
            B, t, D = x.shape
            h = b.norm_attn(x)
            q, k, v = b.qkv(h).chunk(3, dim=-1)
            q = q.view(B, t, self.H, self.Dh).transpose(1, 2)
            k = k.view(B, t, self.H, self.Dh).transpose(1, 2)
            v = v.view(B, t, self.H, self.Dh).transpose(1, 2)
            if b.use_qk_norm:
                q, k = b.q_norm(q), b.k_norm(k)
            cos, sin = b.rope_cos.index_select(0, posv), b.rope_sin.index_select(0, posv)
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
            self.K[l].index_copy_(2, posv, k.to(self.dt)); self.V[l].index_copy_(2, posv, v.to(self.dt))
            Kc, Vc = self.K[l][:, :, :kvlen], self.V[l][:, :, :kvlen]
            if causal:
                o = F.scaled_dot_product_attention(q, Kc, Vc, is_causal=True)
            elif self.step_backend == SDPBackend.CUDNN_ATTENTION:   # n81: the kernel named, not chosen (see __init__)
                o = torch.ops.aten._scaled_dot_product_cudnn_attention(q, Kc, Vc, mask, False, 0.0, False, False)[0]
            else:
                o = F.scaled_dot_product_attention(q, Kc, Vc, attn_mask=mask)
            o = b.o_proj(o.transpose(1, 2).reshape(B, t, D))
            y = x + o
            z = y if b.ffn is None else y + b._ffn(b.norm_ffn(y))
            if len(m.blocks) > 1 and (l > 0 or use_res) and not m._attn_owns_residual(b):
                z = z + x
            x = z
        if hidden:                                             # n79: the graph prefill picks its position first
            return x
        # n79: a prefill needs the last position's logits only (the head over a 32k prompt was a 2 GB bf16 tensor: OOM)
        return _head(m, x[:, -1:] if last_only else x)

    def _embed(self, ids):
        return self.m.embed(ids) * self.m._embed_scale_factor

    @torch.no_grad()
    def prefill(self, ids):
        """ids: list of token ids, starting from position 0. Returns the last position's logits (fp32)."""
        t = len(ids); assert t <= self.max_len
        if self.graph_prefill and self.use_graphs:
            return self._prefill_graph(ids)
        x = self._embed(torch.tensor([ids], device=self.dev))
        posv = torch.arange(t, device=self.dev)
        lg = self._blocks(x, posv, t, None, causal=True, last_only=True)
        self.pos = t
        return lg[0, -1].float()

    def _prefill_graph(self, ids):
        """n79 (graph_prefill, default off; for a like-for-like prompt-read comparison with v4's graph segments): the
        prompt right-padded to a 256-multiple and replayed from one CUDA graph per padded length. Exact for v4f (causal,
        stateless but for K / V): the pads' K / V land past the prompt, where decode steps overwrite them and the step
        mask hides them; the logits come from position t - 1."""
        t = len(ids); Lb = min(self.max_len, -(-t // self.bucket) * self.bucket)
        if Lb not in self.pgraphs:
            ids_b = torch.zeros(1, Lb, dtype=torch.long, device=self.dev); last = torch.zeros(1, dtype=torch.long, device=self.dev)
            posv = torch.arange(Lb, device=self.dev)
            blocks = self._read_blocks()

            def body():
                x = blocks(self._embed(ids_b), posv, Lb, None, causal=True, hidden=True)
                return _head(self.m, x.index_select(1, last))[0, -1]
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2):
                    body()
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                out = body()
            self.pool = g.pool(); self.pgraphs[Lb] = (g, ids_b, last, posv, out)   # posv too: the graph reads it
        g, ids_b, last, _, out = self.pgraphs[Lb]
        ids_b.zero_(); ids_b[0, :t].copy_(torch.tensor(ids)); last.fill_(t - 1)
        g.replay()
        self.pos = t
        return out.float() if out.dtype != torch.float32 else out.clone()

    def _read_blocks(self):
        """n81 10-26 (Glenn: compile the prompt read of BOTH models): _blocks for the graph prefill, plain or through
        torch.compile (one compile per padded length; read 1.12-1.17x faster: temp/n81_lever_a.py)."""
        if not self.compile_read:
            return self._blocks
        if getattr(self, "_cread", None) is None:
            import torch._inductor.config as ic
            ic.emulate_precision_casts = True
            self._cread = torch.compile(self._blocks, mode="default", dynamic=False, fullgraph=False)
        return self._cread

    def _step_eager(self, tok, pos, kvlen, mask):
        x = self._embed(tok)
        return self._blocks(x, pos, kvlen, mask, causal=False)[0, -1]

    @torch.no_grad()
    def step(self, token):
        """Feed one token at the current position; returns its next-token logits (fp32)."""
        assert self.pos < self.max_len
        kvlen = min(self.max_len, -(-(self.pos + 1) // self.bucket) * self.bucket)
        if kvlen not in self.s_mask:
            self.s_mask[kvlen] = torch.full((1, 1, 1, kvlen), float("-inf"), device=self.dev, dtype=self.dt)
        mask = self.s_mask[kvlen]
        mask[..., :self.pos + 1] = 0
        mask[..., self.pos + 1:] = float("-inf")
        self.s_tok.fill_(int(token)); self.s_pos.fill_(self.pos)
        body = self._body()
        if self.use_graphs:
            if kvlen not in self.graphs:
                with self._sdpa():                             # the backend is chosen while tracing / capturing
                    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(st):
                        for _ in range(3 if self.compile_step else 2):   # (re-writes this position's K / V: same values)
                            body(self.s_tok, self.s_pos, kvlen, mask)
                    torch.cuda.current_stream().wait_stream(st)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self.pool):
                        out = body(self.s_tok, self.s_pos, kvlen, mask)
                self.pool = g.pool(); self.graphs[kvlen] = (g, out)
            g, out = self.graphs[kvlen]
            g.replay()
            lg = out.float() if out.dtype != torch.float32 else out.clone()   # never the graph's own buffer (V4Decoder)
        else:
            with self._sdpa():
                lg = body(self.s_tok, self.s_pos, kvlen, mask).float()
        self.pos += 1
        return lg

    def _sdpa(self):
        import contextlib
        if self.step_backend is None:
            return contextlib.nullcontext()
        from torch.nn.attention import sdpa_kernel
        return sdpa_kernel([self.step_backend])

    def _body(self):
        """The step, plain or through torch.compile (as V4Decoder._body: mode default -- the CUDA graph is our own --,
        emulate_precision_casts). dynamic=None: the key length (one per 256-bucket) turns symbolic on its second value, so
        the step compiles twice in all, not once per bucket (dynamic=False would also hit the recompile limit)."""
        if not self.compile_step:
            return self._step_eager
        if getattr(self, "_cbody", None) is None:
            import torch._inductor.config as ic
            ic.emulate_precision_casts = True
            self._cbody = torch.compile(self._step_eager, mode="default", fullgraph=False)
        return self._cbody


class V4Decoder:
    """Exact streaming decode for the v4 hybrid (12 chunked top-k attention layers + 1 GDN-2 memory layer). Mirrors the
    full forward (LM.forward -> HybridAttentionBlock.forward_sequence / SpikingHebbianBlock.forward_sequence) token by
    token, carrying every piece of state the full forward builds implicitly:
      attention   per layer a KV cache of the CURRENT aligned chunk only (C = attn_chunk; the full forward folds chunks
                  into the batch, so RoPE positions, keys and the canon conv all RESTART at every C boundary -> the cache
                  is cleared there; memory is bounded by C whatever the context), the canon conv's last k-1 inputs
      memory      LIF membrane (the exact LIF kernel starts from 0, so the carried membrane is folded into the first
                  current exactly: pre_0 = beta * mem + cur_0), previous-token spikes (keys use the PREVIOUS token's
                  spikes), the GDN-2 state M (gdn2_chunked_dense takes and returns it), the canon_a tail, and the HOLA
                  cache (top-W past (key, value, score) entries, refreshed at every C boundary as _hola_read does)
      engram v2   the last max(order)-1 tokens (its n-gram hash left-pads with 0)
    feed(ids) accepts any number of tokens (prefill = one call); segments never cross a chunk boundary.
    window=W (n77 10-24, inference-only): SLIDING-window attention of W tokens instead of aligned chunks (= the full forward
    with attn_chunk 0, attn_window W): every query sees its last W keys (no cliff at chunk starts: NL facts 200 tokens
    back across a boundary 10% -> 95%, temp/n77_nl_recall.py), absolute RoPE positions (table built here up to
    max_len), cache trimmed to the last W-1 keys, canon conv continuous. The memory block is unchanged (HOLA still
    refreshes at every attn_chunk boundary, as in the full forward).
    parity: temp/n77_stream_parity.py (decode logits vs the full forward on the same weights)."""

    def __init__(self, model, window=0, max_len=1 << 17):
        m = self.m = model
        _check_lm(m)
        self.attn = [b for b in m.blocks if hasattr(b, "qkv")]
        Cs = {int(b.attn_chunk) for b in self.attn}
        if window and Cs == {0}:                               # blocks already switched to the sliding window: the
            Cs = {int(getattr(b, "hola_chunk", 0) or 1024) for b in m.blocks if not hasattr(b, "qkv")} or {1024}
        assert len(Cs) == 1 and min(Cs) > 0, f"needs one shared attn_chunk, got {Cs}"   # chunk still sets HOLA refresh
        self.C = Cs.pop()
        for b in self.attn:
            assert b.ffn is not None and (not b.attn_window or b.attn_window == window),                 "no-FFN blocks / a block window different from the decoder's are not supported"
        for b in m.blocks:
            if not hasattr(b, "qkv"):
                bad = [f for f in ("recurrent", "pre_conv", "ff_conv", "use_fla_kernel", "gdn2_loglinear", "local_attn",
                                   "swa_window", "spk_mlp", "return_mem_read", "titans", "sdm_enabled", "osdn_enabled")
                       if getattr(b, f, False)]
                assert not bad and b.gdn2_enabled and b.gdn2_chunked and b.delta_rule and b.lif_exact and b.n_heads == 1 \
                    and getattr(b, "deltaproduct_nh", 1) <= 1, f"memory block flags not supported: {bad}"
                assert not getattr(b, "hola_w", 0) or int(b.hola_chunk) == self.C, "HOLA chunk must equal attn_chunk"
        p = next(m.parameters())
        self.dev, self.dt = p.device, p.dtype
        self.eng_hist = (max(m.engram_v2.orders) - 1) if m.engram_v2_enabled else 0
        self._beta_f = {}                                      # n78: per memory block, the binary LIF kernel's beta
        self.compile_step = False                              # n78: torch.compile the static step body (fusion)
        self.compile_read = False                              # n81: torch.compile the prompt-read segment body too
        self.read_fallback = set()                             # n81: segment keys whose compile failed (plain body)
        self.W = int(window)
        if self.W:
            from hybrid_attn import _rope_freqs                # same function / base as the blocks' own tables
            self.cos, self.sin = _rope_freqs(self.attn[0].d_head, max_len, device=self.dev)
            self.max_len = max_len
        self.reset()

    def reset(self):
        m, dev = self.m, self.dev
        self.pos, self.hist = 0, []
        self.kv = {}                                           # attention layer -> (K, V) of the current chunk
        self.ctail = {}                                        # block index -> last k-1 conv inputs
        self.mem = {}                                          # memory block index -> dict of carried state
        for l, b in enumerate(m.blocks):
            if not hasattr(b, "qkv"):
                self.mem[l] = dict(v=torch.zeros(1, b.n_neurons, device=dev), prev=torch.zeros(1, b.n_neurons, device=dev),
                                   M=torch.zeros(b.n_state_units, b.d_state, b.d_state, device=dev),
                                   hc=None, hcur=[])

    # ------------------------------------------------------------ pieces
    def _conv_tail(self, key, x, w, k):
        """Causal depthwise conv of x (1, t, D) continuing from the stored k-1 previous inputs (zeros at a restart)."""
        tail = self.ctail.get(key)
        if tail is None:
            tail = torch.zeros(1, k - 1, x.shape[-1], device=x.device, dtype=x.dtype)
        xin = torch.cat([tail.to(x.dtype), x], 1)
        self.ctail[key] = xin[:, -(k - 1):].detach()
        return F.conv1d(xin.transpose(1, 2), w.to(x.dtype), groups=x.shape[-1]).transpose(1, 2)

    def _attn_block(self, l, b, x, p):
        """HybridAttentionBlock._block for t new tokens at in-chunk offset p (cache holds positions 0..p-1); in window
        mode p is ignored and positions are absolute (self.pos ..)."""
        B, t, D = x.shape
        if self.W:
            p = self.pos
            assert p + t <= self.max_len, "context longer than the decoder's RoPE table (raise max_len)"
        H, Dh = b.n_heads, b.d_head
        x_a = x + self._conv_tail(("a", l), x, b.canon_w, b.attn_canon) if b.attn_canon else x
        h = b.norm_attn(x_a)
        if "qkv" in b.sc:
            h = b.sc["qkv"](h)
        q, k, v = b.qkv(h).chunk(3, dim=-1)
        q = q.view(B, t, H, Dh).transpose(1, 2); k = k.view(B, t, H, Dh).transpose(1, 2); v = v.view(B, t, H, Dh).transpose(1, 2)
        if b.use_qk_norm:
            q, k = b.q_norm(q), b.k_norm(k)
        cos, sin = (self.cos[p:p + t], self.sin[p:p + t]) if self.W else (b.rope_cos[p:p + t], b.rope_sin[p:p + t])
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        if "q" in b.sc:
            q = b.sc["q"](q)
        if "v" in b.sc:
            v = b.sc["v"](v)
        k0 = p - (self.kv[l][0].shape[2] if l in self.kv else 0)          # absolute position of the first cached key
        if l in self.kv:
            k = torch.cat([self.kv[l][0], k], 2); v = torch.cat([self.kv[l][1], v], 2)
        self.kv[l] = (k[:, :, -(self.W - 1):], v[:, :, -(self.W - 1):]) if self.W else (k, v)
        qpos = torch.arange(p, p + t, device=x.device); kpos = torch.arange(k0, p + t, device=x.device)
        if b.attn_topk:
            o = torch.matmul(b._topk_weights(q, k, qpos=qpos, kpos=kpos, w=self.W).to(v.dtype), v)
        else:
            mask = (kpos[None, :] <= qpos[:, None]) & ((qpos[:, None] - kpos[None, :] < self.W) if self.W else True)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = o.transpose(1, 2).contiguous().view(B, t, D)
        if b.attn_gate is not None:
            o = o * torch.sigmoid(b.attn_gate(h))
        o = b.o_proj(b.sc["o"](o) if "o" in b.sc else o)
        y = x + o
        return y + b._ffn(b.norm_ffn(y))

    def _lif(self, b, cur, st):
        import demo_ops                                          # noqa: F401  registers torch.ops.n78.lif_step
        beta = torch.sigmoid(b.beta_logit) if b.lif_learn_beta else b.beta_val
        beta_t = beta if torch.is_tensor(beta) else torch.full((cur.shape[-1],), float(beta), device=cur.device)
        if id(b) not in self._beta_f:                            # the binary kernel's python-float beta, read once
            self._beta_f[id(b)] = float(beta) if not torch.is_tensor(beta) else float(beta.reshape(-1)[0])
        # n78 10-25: the neuron runs in fp32 whatever the model dtype -- the kernel returns the membrane in its INPUT
        # dtype, so a bf16 decoder rounded the carried membrane to bf16 between every two tokens (the full forward keeps
        # it fp32 inside the kernel across the whole sequence): bf16 decoder ppl on text repeated beyond the window 42.4
        # -> 27.2 (bf16 full forward 25.8; temp/n78_memfp32_test.py). Spikes are integers: exact back in bf16.
        dt = cur.dtype
        cur = cur.to(torch.float32, copy=True)
        cur[:, 0] = cur[:, 0] + beta_t.float() * st["v"].float()   # carried membrane, folded in exactly
        spk, mem = torch.ops.n78.lif_step(cur, beta_t.float(), self._beta_f[id(b)], bool(b.lif_ml), float(b.lif_reset),
                                          int(b.lif_levels), bool(b.lif_signed),
                                          float(b.lif_alpha) if b.lif_alpha is not None else 2.0)
        st["v"] = mem.detach()
        return spk.to(dt)

    def _hola(self, b, q, k, v, score, st, p):
        """_hola_read for t new tokens at in-chunk offset p of global chunk j = pos // C (state: cache of top-W past
        entries hc = (kn, v, score), and the current chunk's candidates hcur)."""
        t = q.shape[1]; D = q.shape[-1]
        rms = lambda z: z * torch.rsqrt(z.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(z.dtype)
        qn, kn = rms(q) * b.hola_gq.to(q.dtype), rms(k) * b.hola_gk.to(k.dtype)
        st["hcur"].append((kn, v, score))
        if st["hc"] is None:                                   # first chunk: no cache yet -> zero read
            return torch.zeros_like(v)
        Kc, Vc, _ = st["hc"]
        sink = b.hola_sink.to(q.dtype)
        logits = torch.cat([torch.bmm(qn, Kc.transpose(1, 2)), (qn * sink).sum(-1, keepdim=True)], -1) * D ** -0.5
        att = torch.softmax(logits.float(), dim=-1).to(v.dtype)[..., :-1]
        return torch.bmm(att, Vc) * b.hola_lam.to(v.dtype)

    def _hola_boundary(self, b, st):
        """At a chunk boundary: cache <- top-W of (cache + the finished chunk's entries), as _hola_read's loop."""
        kn = torch.cat([e[0] for e in st["hcur"]], 1); v = torch.cat([e[1] for e in st["hcur"]], 1)
        sc = torch.cat([e[2] for e in st["hcur"]], 1)
        if st["hc"] is not None:
            kn, v, sc = torch.cat([st["hc"][0], kn], 1), torch.cat([st["hc"][1], v], 1), torch.cat([st["hc"][2], sc], 1)
        top = sc.topk(min(int(b.hola_w), sc.shape[1]), dim=1)
        g = top.indices.unsqueeze(-1).expand(-1, -1, kn.shape[-1])
        st["hc"] = (kn.gather(1, g), v.gather(1, g), top.values); st["hcur"] = []

    def _mem_block(self, l, b, x, p, ablate):
        st = self.mem[l]
        x_cur = x + self._conv_tail(("m", l), x, b.canon_a_w, b.canon_a) if b.canon_a else x
        x_cur = x_cur if b.pre_norm is None else b.pre_norm(x_cur)
        sc = getattr(b, "sc_in", None)
        cur = b.to_current(sc(x_cur) if sc is not None else x_cur)
        spk = self._lif(b, cur, st)
        prev = torch.cat([st["prev"].unsqueeze(1).to(spk.dtype), spk[:, :-1]], 1)
        st["prev"] = spk[:, -1].detach()
        k, v, q = b.W_k(prev), b.W_v(spk), b.W_q(spk)
        ff = b.W_ff(spk)
        if b.write_gate:
            v = v * torch.sigmoid(b.W_gate(spk))
        if ablate:
            return b.norm(ff) if not b.spk_layerscale else b.norm(ff) * b.ls_gamma.to(ff.dtype)
        from delta_chunked import gdn2_chunked_dense
        bl, wl, fl = b.W_b(spk), b.W_w(spk), b.W_fdecay(spk)
        if b.gdn2_gate_norm:
            bl, wl, fl = b.gate_norm_b(bl), b.gate_norm_w(wl), b.gate_norm_f(fl) + b.gate_bias_f
        bs, ws = torch.sigmoid(bl), b.beta_floor + (1 - b.beta_floor) * torch.sigmoid(wl)
        al = torch.exp(-torch.exp(b.gdn2_a) * F.softplus(fl))
        if getattr(b, "hola_w", 0):
            r, st["M"], u = gdn2_chunked_dense(v, k, q, bs, ws, al, st["M"], chunk=b.gdn2_chunk_size, want_u=True)
            r = r + self._hola(b, q, k, v, u, st, p)
        else:
            r, st["M"] = gdn2_chunked_dense(v, k, q, bs, ws, al, st["M"], chunk=b.gdn2_chunk_size)
        if b.mem_out_gate:
            r = r * (2.0 * torch.sigmoid(b.W_og(spk)))
        out = b.norm(r + ff)
        return out * b.ls_gamma.to(out.dtype) if b.spk_layerscale else out

    # ------------------------------------------------------------ driver
    @torch.no_grad()
    def _segment(self, ids, ablate, last_only=False):
        m = self.m
        p = self.pos % self.C
        if p == 0 and self.pos > 0:                            # chunk boundary: attention + canon restart, HOLA refresh
            if not self.W:                                     # (window mode: attention and canon run on unbroken)
                self.kv.clear()
                for key in [k for k in self.ctail if k[0] == "a"]:
                    del self.ctail[key]
            for l, b in enumerate(m.blocks):
                if l in self.mem and getattr(b, "hola_w", 0) and not ablate:
                    self._hola_boundary(b, self.mem[l])
        idx = torch.tensor([ids], device=self.dev)
        x = m.embed(idx) * m._embed_scale_factor
        if m.engram_v2_enabled:
            h = self.hist[-self.eng_hist:] if self.eng_hist else []
            full = torch.tensor([[0] * (self.eng_hist - len(h)) + h + ids], device=self.dev)
            x = x + m.engram_v2(full)[:, -len(ids):]
        use_res = (m.d == m.d_mem)
        for l, b in enumerate(m.blocks):
            if hasattr(b, "qkv"):
                z = self._attn_block(l, b, x, p)
            else:
                z = self._mem_block(l, b, x, p, ablate)
            if len(m.blocks) > 1 and (l > 0 or use_res) and not m._attn_owns_residual(b):
                z = z + x
            x = z
        self.pos += len(ids)
        self.hist = (self.hist + ids)[-max(self.eng_hist, 1):]
        return _head(m, x[:, -1:] if last_only else x)        # n79: a prompt read needs only the last logits

    @torch.no_grad()
    def feed(self, ids, ablate=False, all_logits=False):
        """Feed tokens (any number, split at chunk boundaries). Returns the last position's logits (fp32), or every
        position's (t, V) logits with all_logits."""
        outs = []
        while ids:
            n = min(len(ids), self.C - self.pos % self.C)
            lg = self._segment(ids[:n], ablate, last_only=not all_logits)
            outs.append(lg[0] if all_logits else lg[0, -1:])
            ids = ids[n:]
        out = torch.cat(outs, 0).float()
        return out if all_logits else out[-1]

    # ------------------------------------------------------------ static-buffer single-token step (CUDA graphs)
    # The eager path above is the reference (prefill, parity). For decode speed, step() runs ONE token on fixed buffers
    # (K/V slots, conv tails, LIF / spike / M state, HOLA candidates) updated in place, so the whole step can be captured
    # once per key-length bucket and replayed. load_static() copies the eager state in after a feed(); unload_static()
    # copies it back (another feed(), save / resume). Same math as _segment with t = 1.
    def load_static(self, graphs=True, bucket=256):
        m, dev, dt = self.m, self.dev, self.dt
        S = self.W or self.C
        self.g_on, self.bucket, self.S = graphs and dev.type == "cuda", bucket, S
        if getattr(self, "sb", None) is None:
            b0 = self.attn[0]
            sb = self.sb = dict(K={}, V={}, kpos={}, tail={}, mem={})
            for l, b in enumerate(m.blocks):
                if hasattr(b, "qkv"):
                    sb["K"][l] = torch.zeros(1, b.n_heads, S, b.d_head, device=dev, dtype=dt)
                    sb["V"][l] = torch.zeros_like(sb["K"][l])
                    sb["kpos"][l] = (torch.arange(S, device=dev) if not self.W
                                     else torch.full((S,), -(1 << 40), dtype=torch.long, device=dev))
                    if b.attn_canon:
                        sb["tail"][("a", l)] = torch.zeros(1, b.attn_canon - 1, m.d_mem, device=dev, dtype=dt)
                else:
                    D = b.d_mem
                    st = sb["mem"][l] = dict(v=torch.zeros(1, b.n_neurons, device=dev), prev=torch.zeros(1, b.n_neurons, device=dev, dtype=dt),
                                             M=torch.zeros(b.n_state_units, b.d_state, b.d_state, device=dev))
                    if getattr(b, "hola_w", 0):
                        W_ = int(b.hola_w)
                        st.update(hK=torch.zeros(1, self.C, D, device=dev, dtype=dt), hV=torch.zeros(1, self.C, D, device=dev, dtype=dt),
                                  hS=torch.zeros(1, self.C, device=dev), cK=torch.zeros(1, W_, D, device=dev, dtype=dt),
                                  cV=torch.zeros(1, W_, D, device=dev, dtype=dt), cS=torch.zeros(1, W_, device=dev))
                    if b.canon_a:
                        sb["tail"][("m", l)] = torch.zeros(1, b.canon_a - 1, m.d_mem, device=dev, dtype=dt)
            sb["tok"] = torch.zeros(1, self.eng_hist + 1, dtype=torch.long, device=dev)
            sb["q"] = torch.zeros(1, dtype=torch.long, device=dev)          # query position (in-chunk, or absolute)
            sb["slot"] = torch.zeros(1, dtype=torch.long, device=dev)       # K/V slot of this token
            sb["hslot"] = torch.zeros(1, dtype=torch.long, device=dev)      # HOLA candidate slot (pos % C)
            sb["abs"] = torch.zeros(1, dtype=torch.long, device=dev)        # absolute position (window kpos)
            sb["cos"], sb["sin"] = (self.cos, self.sin) if self.W else (b0.rope_cos, b0.rope_sin)
            self.graphs, self.pool = {}, None
        sb = self.sb
        p = self.pos % self.C
        with torch.no_grad():
            for l in sb["K"]:
                sb["K"][l].zero_(); sb["V"][l].zero_()
                if self.W:
                    sb["kpos"][l].fill_(-(1 << 40))
                if l in self.kv:
                    k, v = self.kv[l]; n = k.shape[2]
                    if self.W:
                        ps = torch.arange(self.pos - n, self.pos, device=dev); sl = ps % S
                        sb["K"][l].index_copy_(2, sl, k.to(dt)); sb["V"][l].index_copy_(2, sl, v.to(dt)); sb["kpos"][l].index_copy_(0, sl, ps)
                    else:
                        sb["K"][l][:, :, :n] = k; sb["V"][l][:, :, :n] = v
            for key, buf in sb["tail"].items():
                buf.copy_(self.ctail[key]) if key in self.ctail else buf.zero_()
            for l, st in sb["mem"].items():
                e = self.mem[l]
                st["v"].copy_(e["v"]); st["prev"].copy_(e["prev"]); st["M"].copy_(e["M"])
                if "hK" in st:
                    st["hK"].zero_(); st["hV"].zero_(); st["hS"].zero_()
                    if e["hcur"]:
                        kn = torch.cat([x[0] for x in e["hcur"]], 1); vv = torch.cat([x[1] for x in e["hcur"]], 1)
                        ss = torch.cat([x[2] for x in e["hcur"]], 1); n = kn.shape[1]
                        st["hK"][:, :n] = kn; st["hV"][:, :n] = vv; st["hS"][:, :n] = ss
                    st["hola_on"] = e["hc"] is not None
                    if e["hc"] is not None:
                        st["cK"].copy_(e["hc"][0]); st["cV"].copy_(e["hc"][1]); st["cS"].copy_(e["hc"][2])
        self.static = True

    def unload_static(self):
        """Buffers -> eager state (for feed() after step()s, or saving)."""
        sb, S = self.sb, self.S
        p = self.pos % self.C
        if p == 0 and self.pos > 0:                            # at a boundary the finished chunk is still pending (the
            p = self.C                                         # next segment / step clears K/V and merges HOLA)
        for l in sb["K"]:
            if self.W:
                n = min(self.pos, S - 1)
                ps = torch.arange(self.pos - n, self.pos, device=self.dev); sl = ps % S
                self.kv[l] = (sb["K"][l].index_select(2, sl).clone(), sb["V"][l].index_select(2, sl).clone())
            elif p:
                self.kv[l] = (sb["K"][l][:, :, :p].clone(), sb["V"][l][:, :, :p].clone())
            else:
                self.kv.pop(l, None)
        for key, buf in sb["tail"].items():
            self.ctail[key] = buf.clone()
        for l, st in sb["mem"].items():
            e = self.mem[l]
            e["v"], e["prev"], e["M"] = st["v"].clone(), st["prev"].clone(), st["M"].clone()
            if "hK" in st:
                e["hcur"] = [(st["hK"][:, :p].clone(), st["hV"][:, :p].clone(), st["hS"][:, :p].clone())] if p else []
                e["hc"] = (st["cK"].clone(), st["cV"].clone(), st["cS"].clone()) if st["hola_on"] else None
        self.static = False

    def _step_body(self, kvlen, hola_on):
        m, sb = self.m, self.sb
        tok = sb["tok"]
        x = m.embed(tok[:, -1:]) * m._embed_scale_factor
        if m.engram_v2_enabled:
            x = x + m.engram_v2(tok)[:, -1:]
        use_res = (m.d == m.d_mem)

        def conv(key, x, w, k):
            xin = torch.cat([sb["tail"][key].to(x.dtype), x], 1)
            sb["tail"][key].copy_(xin[:, 1:])
            return F.conv1d(xin.transpose(1, 2), w.to(x.dtype), groups=x.shape[-1]).transpose(1, 2)

        cos, sin = sb["cos"].index_select(0, sb["q"]), sb["sin"].index_select(0, sb["q"])
        for l, b in enumerate(m.blocks):
            if hasattr(b, "qkv"):
                H, Dh = b.n_heads, b.d_head
                x_a = x + conv(("a", l), x, b.canon_w, b.attn_canon) if b.attn_canon else x
                h = b.norm_attn(x_a)
                if "qkv" in b.sc:
                    h = b.sc["qkv"](h)
                q, k, v = b.qkv(h).chunk(3, dim=-1)
                q = q.view(1, 1, H, Dh).transpose(1, 2); k = k.view(1, 1, H, Dh).transpose(1, 2); v = v.view(1, 1, H, Dh).transpose(1, 2)
                if b.use_qk_norm:
                    q, k = b.q_norm(q), b.k_norm(k)
                q, k = _rope(q, cos, sin), _rope(k, cos, sin)
                if "q" in b.sc:
                    q = b.sc["q"](q)
                if "v" in b.sc:
                    v = b.sc["v"](v)
                sb["K"][l].index_copy_(2, sb["slot"], k.to(self.dt)); sb["V"][l].index_copy_(2, sb["slot"], v.to(self.dt))
                if self.W:
                    sb["kpos"][l].index_copy_(0, sb["slot"], sb["abs"])
                Kc, Vc, kp = sb["K"][l][:, :, :kvlen], sb["V"][l][:, :, :kvlen], sb["kpos"][l][:kvlen]
                qpos = sb["abs"] if self.W else sb["q"]
                if b.attn_topk:
                    o = torch.matmul(b._topk_weights(q, Kc, qpos=qpos, kpos=kp, w=self.W).to(Vc.dtype), Vc)
                else:
                    d = qpos[:, None] - kp[None, :]
                    o = F.scaled_dot_product_attention(q, Kc, Vc, attn_mask=(d >= 0) & ((d < self.W) if self.W else True))
                o = o.transpose(1, 2).reshape(1, 1, -1)
                if b.attn_gate is not None:
                    o = o * torch.sigmoid(b.attn_gate(h))
                o = b.o_proj(b.sc["o"](o) if "o" in b.sc else o)
                y = x + o
                z = y + b._ffn(b.norm_ffn(y))
            else:
                st = sb["mem"][l]
                x_cur = x + conv(("m", l), x, b.canon_a_w, b.canon_a) if b.canon_a else x
                x_cur = x_cur if b.pre_norm is None else b.pre_norm(x_cur)
                scin = getattr(b, "sc_in", None)
                cur = b.to_current(scin(x_cur) if scin is not None else x_cur)
                e = dict(v=st["v"])
                spk = self._lif(b, cur, e)
                st["v"].copy_(e["v"])
                k = b.W_k(st["prev"].unsqueeze(1).to(spk.dtype))
                st["prev"].copy_(spk[:, -1])
                v, q = b.W_v(spk), b.W_q(spk)
                ff = b.W_ff(spk)
                if b.write_gate:
                    v = v * torch.sigmoid(b.W_gate(spk))
                from delta_chunked import gdn2_chunked_dense
                bl, wl, fl = b.W_b(spk), b.W_w(spk), b.W_fdecay(spk)
                if b.gdn2_gate_norm:
                    bl, wl, fl = b.gate_norm_b(bl), b.gate_norm_w(wl), b.gate_norm_f(fl) + b.gate_bias_f
                bs, ws = torch.sigmoid(bl), b.beta_floor + (1 - b.beta_floor) * torch.sigmoid(wl)
                al = torch.exp(-torch.exp(b.gdn2_a) * F.softplus(fl))
                if "hK" in st:
                    r, Mn, u = gdn2_chunked_dense(v, k, q, bs, ws, al, st["M"], chunk=b.gdn2_chunk_size, want_u=True)
                    rms = lambda z_: z_ * torch.rsqrt(z_.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(z_.dtype)
                    qn, kn = rms(q) * b.hola_gq.to(q.dtype), rms(k) * b.hola_gk.to(k.dtype)
                    st["hK"].index_copy_(1, sb["hslot"], kn.to(self.dt)); st["hV"].index_copy_(1, sb["hslot"], v.to(self.dt))
                    st["hS"].index_copy_(1, sb["hslot"], u.float())
                    if hola_on:
                        Dm = q.shape[-1]
                        lg = torch.cat([torch.bmm(qn, st["cK"].to(qn.dtype).transpose(1, 2)),
                                        (qn * b.hola_sink.to(q.dtype)).sum(-1, keepdim=True)], -1) * Dm ** -0.5
                        att = torch.softmax(lg.float(), dim=-1).to(v.dtype)[..., :-1]
                        r = r + torch.bmm(att, st["cV"].to(v.dtype)) * b.hola_lam.to(v.dtype)
                else:
                    r, Mn = gdn2_chunked_dense(v, k, q, bs, ws, al, st["M"], chunk=b.gdn2_chunk_size)
                st["M"].copy_(Mn)
                if b.mem_out_gate:
                    r = r * (2.0 * torch.sigmoid(b.W_og(spk)))
                z = b.norm(r + ff)
                z = z * b.ls_gamma.to(z.dtype) if b.spk_layerscale else z
            if len(m.blocks) > 1 and (l > 0 or use_res) and not m._attn_owns_residual(b):
                z = z + x
            x = z
        return _head(m, x)[0, -1]

    @torch.no_grad()
    def step(self, token):
        """One token on the static buffers (call load_static() after the prefill feed()). Returns its logits (fp32)."""
        assert getattr(self, "static", False), "call load_static() after feed()"
        sb = self.sb
        p = self.pos % self.C
        self._boundary(p)
        h = self.hist[-self.eng_hist:] if self.eng_hist else []
        # n79 10-25: element fills, not torch.tensor(..., device=): a host -> device copy from pageable memory
        # synchronises the stream, so every step waited for the previous one to finish and the GPU idled while the host
        # launched the next replay (2.54 ms / token wall vs 2.13 ms GPU, logs/n78_fuse_test.log; V4FDecoder.step only
        # fills -> its harness steps queued back to back, v4's did not)
        for i, t_ in enumerate([0] * (self.eng_hist - len(h)) + h + [int(token)]):
            sb["tok"][0, i].fill_(t_)
        qv = self.pos if self.W else p
        sb["q"].fill_(qv); sb["abs"].fill_(self.pos); sb["hslot"].fill_(p)
        sb["slot"].fill_(self.pos % self.S if self.W else p)
        kvlen = self.S if self.W else min(self.S, -(-(p + 1) // self.bucket) * self.bucket)
        hola_on = any(st.get("hola_on", False) for st in sb["mem"].values())
        if self.g_on:
            key = (kvlen, hola_on)
            if key not in self.graphs:
                snap = self._snapshot()                        # warm-up / capture run the body: restore state after
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                body = self._body()
                with torch.cuda.stream(s):
                    for _ in range(2 if self.compile_step else 1):   # compiled: compile + one clean run before capture
                        body(kvlen, hola_on); self._restore(snap)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool):
                    out = body(kvlen, hola_on)
                self._restore(snap)
                self.pool = g.pool(); self.graphs[key] = (g, out)
            g, out = self.graphs[key]
            g.replay()
            # n78 10-25: always a COPY -- in fp32 .float() is the graph's own output buffer, which the next replay
            # overwrites (a caller keeping step logits saw them all turn into the last step's: "fp32 static broken")
            lg = out.float() if out.dtype != torch.float32 else out.clone()
        else:
            lg = self._body()(kvlen, hola_on).float()
        self.pos += 1
        self.hist = (self.hist + [int(token)])[-max(self.eng_hist, 1):]
        return lg

    def _body(self):
        """The step body, plain or (compile_step) through torch.compile: Inductor fuses the step's ~1,600 small copy /
        elementwise / reduction kernels (logs/n78_fuse_prof.log); the LIF and sparse-GEMV Triton kernels are opaque
        custom ops (demo_ops.py), so the step stays one graph. mode default (NOT reduce-overhead: the CUDA graph here
        is our own); emulate_precision_casts keeps the bf16 <-> fp32 casts Inductor would otherwise fuse away."""
        if not self.compile_step:
            return self._step_body
        if getattr(self, "_cbody", None) is None:
            import torch._inductor.config as ic
            ic.emulate_precision_casts = True
            self._cbody = torch.compile(self._step_body, mode="default", dynamic=False, fullgraph=False)
        return self._cbody

    def _snapshot(self):
        sb = self.sb
        ts = [*sb["K"].values(), *sb["V"].values(), *sb["kpos"].values(), *sb["tail"].values()]
        for st in sb["mem"].values():
            ts += [t_ for k_, t_ in st.items() if torch.is_tensor(t_)]
        return [(t_, t_.clone()) for t_ in ts]

    def _restore(self, snap):
        for t_, c_ in snap:
            t_.copy_(c_)

    def _boundary(self, p):
        """Chunk-boundary work of the static path (eager, outside the graphs): attention + canon restart (chunk mode),
        HOLA cache <- top-W of (cache + the finished chunk's candidates), as _hola_boundary."""
        if not (p == 0 and self.pos > 0):
            return
        sb = self.sb
        if not self.W:
            for key, buf in sb["tail"].items():
                if key[0] == "a":
                    buf.zero_()
        for l, st in sb["mem"].items():
            if "hK" in st:
                b = self.m.blocks[l]
                kn, v, sc = st["hK"], st["hV"], st["hS"]
                if st["hola_on"]:
                    kn, v, sc = torch.cat([st["cK"], kn], 1), torch.cat([st["cV"], v], 1), torch.cat([st["cS"], sc], 1)
                top = sc.topk(int(b.hola_w), dim=1)
                g = top.indices.unsqueeze(-1).expand(-1, -1, kn.shape[-1])
                st["cK"].copy_(kn.gather(1, g)); st["cV"].copy_(v.gather(1, g)); st["cS"].copy_(top.values)
                st["hola_on"] = True

    # ------------------------------------------------------------ static-buffer prompt read (CUDA-graph segments)
    # n79 10-25 (Glenn: wire it in; prototype temp/n79_graph_feed.py, logs/n79_prefill.log): feed_static() reads any
    # number of tokens ON the step's static buffers, cut (never across a chunk boundary) into the largest SEG_SIZES that
    # fit, each one CUDA-graph replay of _seg_body; fewer than 4 left -> step()s. Padding is impossible (pads would
    # advance the memory state), hence fixed sizes. The eager feed() read was host-bound (~110 ms whatever the length);
    # instant energy per prompt token 45.0 -> 27.4 mJ at 199 tokens, 22.9 -> 12.3 at 512 (5080). One segment = the eager
    # segment exactly (max|diff| 0, fp32 and bf16); other cuts = float-association noise. Window mode only.
    def feed_static(self, ids):
        """Read `ids` onto the static buffers (after load_static(); continues after step()s too). Returns the last
        position's logits (fp32)."""
        assert self.W and getattr(self, "static", False), "window mode, after load_static()"
        ids, lg = list(ids), None
        # n80 10-26 (review fix a): the whole read's ids (+ the engram history before them) go to the GPU ONCE, from
        # pinned memory without a sync; each segment then copies its slice device-to-device (was a pageable host copy per
        # segment = a stream sync each, as the n79 step fix). Same values -> same logits.
        h = self.hist[-self.eng_hist:] if self.eng_hist else []
        eh = self.eng_hist
        allv = torch.tensor([0] * (eh - len(h)) + h + ids).pin_memory().to(self.dev, non_blocking=True)
        off = 0
        while ids:
            n = min(len(ids), self.C - self.pos % self.C)
            T = next((s for s in SEG_SIZES if s <= n), 1)
            lg = self.step(ids[0]) if T == 1 else self._seg(ids[:T], allv[off:off + eh + T])
            ids, off = ids[T:], off + T
        return lg

    def warm_reads(self):
        """n81: capture every segment graph a read can reach -- each SEG_SIZE that fits at each ring length (pos 0, 1,
        257, 513, 769) and with the HOLA cache on (pos C + 1) -- so no compile lands inside an answer (the n81 demo set
        without it: 20 turns 168.6 s vs 10.2 s). Leaves the decoder reset."""
        for p in (0, 1, 257, 513, 769, self.C + 1):
            self.reset(); self.load_static(graphs=True)
            for _ in range(p):
                self.step(1)
            c = self.save_state()
            for T in SEG_SIZES:
                if T <= self.C - p % self.C:
                    self._seg([1] * T); self.restore_state(c)
        self.reset()

    @torch.no_grad()
    def _seg(self, ids, dev_ids=None):
        T, sb, p = len(ids), self.sb, self.pos % self.C
        self._boundary(p)
        segs = sb.setdefault("seg", {})
        if T not in segs:
            segs[T] = dict(tok=torch.zeros(1, self.eng_hist + T, dtype=torch.long, device=self.dev),
                           ar=torch.arange(T, device=self.dev))
        if dev_ids is not None:                                # n80: this segment's slice, already on the GPU
            segs[T]["tok"][0].copy_(dev_ids)
        else:
            h = self.hist[-self.eng_hist:] if self.eng_hist else []
            segs[T]["tok"].copy_(torch.tensor([[0] * (self.eng_hist - len(h)) + h + list(ids)]))
        sb["abs"].fill_(self.pos); sb["hslot"].fill_(p)
        hola_on = any(st.get("hola_on", False) for st in sb["mem"].values())
        # ring keys the segment can see: until the window fills only slots < pos hold keys (slot = pos % S) -> read the
        # first R slots (R a multiple of RING_STEP; 0 for a fresh prompt = exactly the eager feed's attention)
        R = self.S if self.pos >= self.S else -(-self.pos // RING_STEP) * RING_STEP
        key = ("seg", T, hola_on, R)
        body = self._seg_body if key in self.read_fallback else self._seg_fn()
        if not self.g_on:                                      # no CUDA graphs: the same body, eager
            out = body(T, hola_on, R)
            self.pos += T
            self.hist = (self.hist + list(ids))[-max(self.eng_hist, 1):]
            return out.float()
        if key not in self.graphs:
            snap = self._snapshot()
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                try:
                    for _ in range(2 if body is not self._seg_body else 1):   # compiled: compile + one clean run
                        body(T, hola_on, R); self._restore(snap)
                except Exception as e:                         # n81: Inductor fails on some shapes (T 256 in torch
                    if body is self._seg_body:                 # 2.11: a Triton codegen TypeError) -> that key runs the
                        raise                                  # plain body (exact eager math), the others stay compiled
                    print(f"[stream_decode] compiled read failed for {key} ({type(e).__name__}): plain body", flush=True)
                    self._restore(snap); self.read_fallback.add(key); body = self._seg_body
                    body(T, hola_on, R); self._restore(snap)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                out = body(T, hola_on, R)
            self._restore(snap)
            self.pool = g.pool(); self.graphs[key] = (g, out)
        g, out = self.graphs[key]
        g.replay()
        self.pos += T
        self.hist = (self.hist + list(ids))[-max(self.eng_hist, 1):]
        return out.float() if out.dtype != torch.float32 else out.clone()

    def _seg_fn(self):
        """n81 10-26 (Glenn: compile the prompt read of BOTH models): the segment body, plain or through torch.compile
        (as _body; one compile per (size, HOLA on, ring length), ~1 min each on first use; read 1.34-1.43x faster, logits
        at v4's usual float-noise level: temp/n81_lever_a*.py, logs/n81_lever_a.log)."""
        if not self.compile_read:
            return self._seg_body
        if getattr(self, "_cseg", None) is None:
            import torch._inductor.config as ic
            import torch._dynamo
            ic.emulate_precision_casts = True
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            self._cseg = torch.compile(self._seg_body, mode="default", dynamic=False, fullgraph=False)
        return self._cseg

    def _seg_body(self, T, hola_on, R):
        """_step_body for T tokens (window mode); same math as _segment's eager blocks. Every tensor the graph reads
        lives in self.sb (a graph input freed after capture would be read from reused memory)."""
        m, sb = self.m, self.sb
        tok, ar = sb["seg"][T]["tok"], sb["seg"][T]["ar"]
        x = m.embed(tok[:, -T:]) * m._embed_scale_factor
        if m.engram_v2_enabled:
            x = x + m.engram_v2(tok)[:, -T:]
        use_res = (m.d == m.d_mem)

        def conv(key, x, w, k):
            xin = torch.cat([sb["tail"][key].to(x.dtype), x], 1)
            sb["tail"][key].copy_(xin[:, T:])
            return F.conv1d(xin.transpose(1, 2), w.to(x.dtype), groups=x.shape[-1]).transpose(1, 2)

        qpos = sb["abs"] + ar                                  # absolute positions of the segment
        slots = qpos % self.S
        cos, sin = sb["cos"].index_select(0, qpos), sb["sin"].index_select(0, qpos)
        for l, b in enumerate(m.blocks):
            if hasattr(b, "qkv"):
                H, Dh = b.n_heads, b.d_head
                x_a = x + conv(("a", l), x, b.canon_w, b.attn_canon) if b.attn_canon else x
                h = b.norm_attn(x_a)
                if "qkv" in b.sc:
                    h = b.sc["qkv"](h)
                q, k, v = b.qkv(h).chunk(3, dim=-1)
                q = q.view(1, T, H, Dh).transpose(1, 2); k = k.view(1, T, H, Dh).transpose(1, 2); v = v.view(1, T, H, Dh).transpose(1, 2)
                if b.use_qk_norm:
                    q, k = b.q_norm(q), b.k_norm(k)
                q, k = _rope(q, cos, sin), _rope(k, cos, sin)
                if "q" in b.sc:
                    q = b.sc["q"](q)
                if "v" in b.sc:
                    v = b.sc["v"](v)
                Ka = torch.cat([sb["K"][l][:, :, :R], k.to(self.dt)], 2); Va = torch.cat([sb["V"][l][:, :, :R], v.to(self.dt)], 2)
                kp = torch.cat([sb["kpos"][l][:R], qpos])
                if b.attn_topk:
                    o = torch.matmul(b._topk_weights(q, Ka, qpos=qpos, kpos=kp, w=self.W).to(Va.dtype), Va)
                else:
                    d = qpos[:, None] - kp[None, :]
                    o = F.scaled_dot_product_attention(q, Ka, Va, attn_mask=(d >= 0) & (d < self.W))
                sb["K"][l].index_copy_(2, slots, k.to(self.dt)); sb["V"][l].index_copy_(2, slots, v.to(self.dt))
                sb["kpos"][l].index_copy_(0, slots, qpos)
                o = o.transpose(1, 2).reshape(1, T, -1)
                if b.attn_gate is not None:
                    o = o * torch.sigmoid(b.attn_gate(h))
                o = b.o_proj(b.sc["o"](o) if "o" in b.sc else o)
                y = x + o
                z = y + b._ffn(b.norm_ffn(y))
            else:
                st = sb["mem"][l]
                x_cur = x + conv(("m", l), x, b.canon_a_w, b.canon_a) if b.canon_a else x
                x_cur = x_cur if b.pre_norm is None else b.pre_norm(x_cur)
                scin = getattr(b, "sc_in", None)
                cur = b.to_current(scin(x_cur) if scin is not None else x_cur)
                e = dict(v=st["v"])
                spk = self._lif(b, cur, e)
                st["v"].copy_(e["v"])
                prev = torch.cat([st["prev"].unsqueeze(1).to(spk.dtype), spk[:, :-1]], 1)
                st["prev"].copy_(spk[:, -1])
                k, v, q = b.W_k(prev), b.W_v(spk), b.W_q(spk)
                ff = b.W_ff(spk)
                if b.write_gate:
                    v = v * torch.sigmoid(b.W_gate(spk))
                from delta_chunked import gdn2_chunked_dense
                bl, wl, fl = b.W_b(spk), b.W_w(spk), b.W_fdecay(spk)
                if b.gdn2_gate_norm:
                    bl, wl, fl = b.gate_norm_b(bl), b.gate_norm_w(wl), b.gate_norm_f(fl) + b.gate_bias_f
                bs, ws = torch.sigmoid(bl), b.beta_floor + (1 - b.beta_floor) * torch.sigmoid(wl)
                al = torch.exp(-torch.exp(b.gdn2_a) * F.softplus(fl))
                if "hK" in st:
                    r, Mn, u = gdn2_chunked_dense(v, k, q, bs, ws, al, st["M"], chunk=b.gdn2_chunk_size, want_u=True)
                    rms = lambda z_: z_ * torch.rsqrt(z_.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(z_.dtype)
                    qn, kn = rms(q) * b.hola_gq.to(q.dtype), rms(k) * b.hola_gk.to(k.dtype)
                    hs = sb["hslot"] + ar
                    st["hK"].index_copy_(1, hs, kn.to(self.dt)); st["hV"].index_copy_(1, hs, v.to(self.dt))
                    st["hS"].index_copy_(1, hs, u.float())
                    if hola_on:
                        Dm = q.shape[-1]
                        lg = torch.cat([torch.bmm(qn, st["cK"].to(qn.dtype).transpose(1, 2)),
                                        (qn * b.hola_sink.to(q.dtype)).sum(-1, keepdim=True)], -1) * Dm ** -0.5
                        att = torch.softmax(lg.float(), dim=-1).to(v.dtype)[..., :-1]
                        r = r + torch.bmm(att, st["cV"].to(v.dtype)) * b.hola_lam.to(v.dtype)
                else:
                    r, Mn = gdn2_chunked_dense(v, k, q, bs, ws, al, st["M"], chunk=b.gdn2_chunk_size)
                st["M"].copy_(Mn)
                if b.mem_out_gate:
                    r = r * (2.0 * torch.sigmoid(b.W_og(spk)))
                z = b.norm(r + ff)
                z = z * b.ls_gamma.to(z.dtype) if b.spk_layerscale else z
            if len(m.blocks) > 1 and (l > 0 or use_res) and not m._attn_owns_residual(b):
                z = z + x
            x = z
        return _head(m, x[:, -1:])[0, -1]                      # a prompt read needs the last position's logits only

    def save_state(self):
        """n79: a copy of the static decoder state (prefix-state cache): buffers + position + history + HOLA flags."""
        return dict(sb=id(self.sb), snap=self._snapshot(), pos=self.pos, hist=list(self.hist),
                    hola={l: st.get("hola_on", False) for l, st in self.sb["mem"].items()})

    def restore_state(self, s):
        """Back to a save_state() copy (same static buffers; the step / segment graphs stay valid)."""
        assert s["sb"] == id(self.sb) and getattr(self, "static", False), "state saved on other static buffers"
        self._restore(s["snap"]); self.pos, self.hist = s["pos"], list(s["hist"])
        for l, v in s["hola"].items():
            self.sb["mem"][l]["hola_on"] = v
