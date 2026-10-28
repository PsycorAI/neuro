"""demo/engine_stream.py -- 2026-10-24 (Neuro 78, demo plan v2 W1/W2). The demo Engine with v4 STREAMING decode and the
inference-only SLIDING attention window (n77: NL facts across a chunk boundary 10% -> 95%).

  * every attention block runs attn_chunk 0 / attn_window W (default 1024), so the traced full-forward passes (brain
    view, meter) and the streaming decoder compute the same model; the meter's attention context = min(pos + 1, W).
  * untraced decode steps (memory on) use stream_decode.V4Decoder: one eager prefill feed() of the context, then
    load_static() + step() = one CUDA-graph token step on static buffers (static == eager decoder exactly, ppl ==
    full forward: logs/n77_queue6.log). The decoder state is keyed by the token context it holds: a context that
    extends it by a few tokens (grammar-forced tokens) is stepped through, anything else re-prefills.
  * memory-off (ablate) passes keep the parent's full-forward graph path (the decoder's static step has no ablate).
  * a brain switch changes the weights in place -> the decoder state is dropped (graphs stay valid: same storage).
  Engine(...) arguments as engine.Engine, plus window=W."""
import os
import time
import torch
from engine import Engine
from stream_decode import V4Decoder, strip_spike_code_stats

STEP_MAX = 64          # at most this many new tokens are stepped through; more -> re-prefill


class _DevTable:
    """grammar.TokenTable with the per-node masks kept ON the device (the table copied 32k bool + 32k int64 to the GPU
    per decode step, ~5 ms: logs/n78_host_profile.log); the next-node table stays on the CPU, where the engine reads
    it (int(nxt[tok])) -- a GPU copy there cost a sync per token. Same values."""

    def __init__(self, table):
        self.t, self.dev = table, {}

    def __getattr__(self, k):
        return getattr(self.t, k)

    def mask(self, grammar, node, device="cpu"):
        m, nx = self.t.mask(grammar, node)
        if str(device) == "cpu":
            return m, nx
        key = (id(m), str(device))
        if key not in self.dev:
            self.dev[key] = (m, m.to(device))                  # keep m referenced: its id stays unique
        return self.dev[key][1], nx


class StreamEngine(Engine):
    def __init__(self, run, device="cuda", window=1024, dtype="fp32", **kw):
        super().__init__(run, device, **kw)
        self.W = int(window)
        if dtype == "bf16" and device.startswith("cuda"):
            # n78 10-25 (Glenn: yes): bf16 weights (half the bytes per decode step) + bf16 autocast around every full
            # forward (traced passes, memory-off graphs, memory test) = the trained regime (val ppl 18.47 vs fp32 18.49,
            # temp/n78_bf16_full.py; full forward with bf16 weights needs autocast). The decoder runs the bf16 modules
            # directly (stream_decode handles the dtypes). Brains merge into the bf16 weights (restored from copies).
            self.model.to(torch.bfloat16)
            fwd = self.model.forward

            def forward_bf16(*a, **k):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return fwd(*a, **k)
            self.model.forward = forward_bf16
        self.dtype = dtype if device.startswith("cuda") else "fp32"
        for b in self.model.blocks:
            if hasattr(b, "qkv"):
                b.attn_chunk, b.attn_window = 0, self.W
        self.tracer.W = self.W
        strip_spike_code_stats(self.model)                     # telemetry-only stats off (bit-identical outputs)
        # n78 W3 (Glenn 10-25): decode steps read only the weight rows of NONZERO inputs in the Linears whose inputs are
        # mostly exact zeros (FFN down, attention output, memory block; sparse_gemv.py). Exact up to summation order.
        # DEMO_SPARSE=0 = the dense cuBLAS path.
        self.sparse = 0
        if device.startswith("cuda") and os.environ.get("DEMO_SPARSE", "1") != "0":
            import sparse_gemv
            self.sparse = sparse_gemv.enable(self.model, sparse_gemv.demo_pred)
        # n78: positions far past block_size (the stream continues past the 2048 cut; RoPE table 65536 x d_head x 2)
        self.dec = V4Decoder(self.model, window=self.W, max_len=65536)
        # n78 10-25 (Glenn: fuse the tiny kernels): the step body through torch.compile (1,772 -> 391 kernels, 5.05 ->
        # 2.54 ms / token, val ppl equal; logs/n78_fuse_test.log). First use compiles (~2 min, during warm-up).
        # DEMO_COMPILE=0 = the plain step.
        self.dec.compile_step = device.startswith("cuda") and os.environ.get("DEMO_COMPILE", "1") != "0"
        # n81 10-26 (Glenn: compile the prompt read of both models, as the table measures it): the graph-segment read
        # body through torch.compile too (40 segment shapes, all compiled in warm-up). OFF by default (Glenn 10-26: the
        # ~30 min warm-up is too long for the ~20% prompt-read saving); DEMO_COMPILE_READ=1 = on.
        self.dec.compile_read = self.dec.compile_step and os.environ.get("DEMO_COMPILE_READ", "0") == "1"
        # n79 10-25 (Glenn: wire it in): prompt reads as CUDA-graph segments + the tools-block state cache (_read)
        self.graph_feed = device.startswith("cuda") and os.environ.get("DEMO_GRAPH_FEED", "1") != "0"
        self.prefix_cache = os.environ.get("DEMO_PREFIX_CACHE", "1") != "0"
        self._pfx_ids, self._pfx_cache, self.pfx_hits = self._tools_prefix(), {}, 0
        self._sctx, self._sbrain = None, None
        self.stream = True                                     # False = the parent's full-forward graphs (parity tests)
        self.table = _DevTable(self.table)
        self.trace_step0 = False                               # no traced step-0 pass when it traces nothing
        self.defer_prompt_trace = True                         # prompt traces from the post-answer replay (Glenn 10-25)
        self.graph_trace = True                                # traced passes from CUDA graphs (_traced_logits)
        self.tgraphs, self.tpool = {}, None
        self.phases = []                                       # n78 W4: timed GPU segments of the current request
        self.info.update(attn_window=self.W, stream_decode=True, dtype=self.dtype, sparse_linears=self.sparse,
                         compiled_step=self.dec.compile_step, graph_prompt_read=self.graph_feed,
                         compiled_prompt_read=self.dec.compile_read and self.graph_feed,
                         prefix_state_cache=self.prefix_cache and self.graph_feed)
        self.v4f = self._load_v4f()
        orig = self.tracer.per_position

        def per_position(t0, _orig=orig):                      # + the v4f box's fields (active_f, h0_f) per word
            res, fb = _orig(t0), getattr(self.tracer, "fbox", None)
            if fb is not None:
                act, grid = fb[0][t0:t0 + len(res)].tolist(), fb[1][t0:t0 + len(res)].tolist()
                for r, a_, g_ in zip(res, act, grid):
                    r["active_f"], r["h0_f"] = round(a_, 6), [round(v, 3) for v in g_]
            return res
        self.tracer.per_position = per_position

    def _load_v4f(self):
        """n80 (Glenn 10-25, overnight item 2): the v4f transformer for the page's second "this word" box. Its neurons =
        the FFN hidden units (the input of each SwiGLU down projection, 13 x 9,472), counted as v4's are: active = exactly
        nonzero. Measured offline (temp/n80_v4f_zero_share.py, 13,312 words, both v4f checkpoints): 100.000% per word
        (min 99.999%) -- SwiGLU's silu(w1 x) * w3 x is almost never exactly zero. It runs over the same tokens as every
        traced pass (bf16, eager), inside the timed viz segment, so its energy is not the answer's. +2.2 GB GPU memory.
        DEMO_V4F_BOX=0 = off; =<run> = another checkpoint (default: the chat v4f, same tokenizer)."""
        run = os.environ.get("DEMO_V4F_BOX", "n74d_sft_v4f_chat")
        if run == "0" or not self.device.startswith("cuda"):
            return None
        from engine import latest
        from train_gpu import build_model
        from hybrid_attn import SwiGLU
        try:
            s = torch.load(latest(run), map_location="cpu", weights_only=False, mmap=True)
        except FileNotFoundError as e:
            print(f"[engine_stream] v4f box off: {e}", flush=True)
            return None
        c = dict(s["cfg"])
        if c["vocab"] != self.c["vocab"] or c.get("lut_subword_fallback") != self.c.get("lut_subword_fallback"):
            print(f"[engine_stream] v4f box off: {run} has another tokenizer", flush=True)
            return None
        with torch.device(self.device):
            m = build_model(c)
        miss, unexp = m.load_state_dict(s["model"], strict=False)
        del s
        if miss or unexp:
            print(f"[engine_stream] v4f box off: {run} checkpoint / config mismatch", flush=True)
            return None
        m = m.to(torch.bfloat16).eval()
        m.head = torch.nn.Identity()                           # n80 review fix b: the box needs the FFNs, not logits
        self.fgraphs, self.fpool = {}, None
        ffns = [f for f in m.modules() if isinstance(f, SwiGLU)]
        self._fz = []

        def hook(mod, inp):                                    # per word: nonzero units, and 128 bins of each layer
            z = (inp[0][0] != 0).float()                       # (T, hidden)
            self._fz.append((z.sum(-1), torch.nn.functional.adaptive_avg_pool1d(z[:, None], 128)[:, 0]))
        for f in ffns:
            f.w2.register_forward_pre_hook(hook)
        self.v4f_neurons = sum(f.w2.in_features for f in ffns)
        self.info.update(v4f_box=dict(run=run, neurons=self.v4f_neurons, layers=len(ffns)))
        print(f"[engine_stream] v4f box: {run}, {self.v4f_neurons:,} FFN neurons", flush=True)
        return m

    @torch.no_grad()
    def _v4f_trace(self, ctx):
        """The v4f box's per-word values for ctx: share of FFN neurons exactly nonzero, and the 128-dot grid (dot i =
        the i-th 1/128 of every layer's units, averaged over the layers).
        n80 review fix b: replayed from a CUDA graph of a right-padded forward, one per length bucket (as v4's traced
        pass, _traced_impl; causal: the padding cannot reach the real positions, which are the only ones read). The
        hooks' outputs are captured with the graph and refilled by each replay. Capture failure -> eager for good."""
        T = len(ctx)
        if self.use_graph and self.fgraphs is not None:
            L = min(int(self.c["block_size"]), -(-T // self.bucket) * self.bucket)
            if T <= L:
                if L not in self.fgraphs:
                    static = torch.zeros(1, L, dtype=torch.long, device=self.device)
                    try:
                        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(st):
                            for _ in range(2):
                                self._fz = []; self.v4f(static)
                        torch.cuda.current_stream().wait_stream(st)
                        g = torch.cuda.CUDAGraph()
                        self._fz = []
                        with torch.cuda.graph(g, pool=self.fpool):  # the reductions inside: refreshed by every replay
                            self.v4f(static)
                            cnt = sum(c for c, _ in self._fz); grid = sum(g_ for _, g_ in self._fz) / len(self._fz)
                        self.fpool = g.pool()
                        self.fgraphs[L] = (g, static, cnt, grid)
                    except Exception as e:
                        print(f"[engine_stream] v4f box graph capture failed at L={L}: {type(e).__name__}: {str(e)[:160]}; "
                              f"eager", flush=True)
                        self.fgraphs = None
                if self.fgraphs is not None:
                    g, static, cnt, grid = self.fgraphs[L]
                    static.zero_(); static[0, :T] = torch.tensor(ctx, device=self.device)
                    g.replay()
                    self.tracer.fbox = (cnt[:T].cpu() / self.v4f_neurons, grid[:T].cpu())
                    self._fz = []
                    return
        self._fz = []
        self.v4f(torch.tensor([ctx], device=self.device))
        n = len(self._fz)
        self.tracer.fbox = (sum(c for c, _ in self._fz).cpu() / self.v4f_neurons, (sum(g for _, g in self._fz) / n).cpu())
        self._fz = []

    def _drop(self):
        self._sctx = None

    def _tools_prefix(self):
        """n79: the token ids every decide prompt starts with = the tools block, cut where its encoding stops agreeing
        with whole prompts' (with and without the chat-history block: the tokenizer may merge across the block's end)."""
        from sft_data import render_conversation
        block = self.tool_data.render_tools_block(self.offered, self.tools) + "\n\n"
        a = self.encode(block)
        n = len(a)
        for rest in ("### Instruction:\nhello\n\n### Response:\n", render_conversation([("hi", "hello")])):
            b = self.encode(block + rest)
            n = min(n, next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))))
        return a[:n]

    def _ph(self, kind, t0, n, ctx_len):
        """n78 W4: one timed GPU segment for the measured meter (gpu_meter.py): kind prefill | decode | viz, its token
        count, and the context length (the meter's v4 / v4f reference is looked up at it). Synchronised at the end, so
        the segment holds its own GPU work (the decode loop syncs per token anyway)."""
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.phases.append((kind, t0, time.time(), n, ctx_len))

    @torch.no_grad()
    def _graph_logits(self, ctx, ablate):
        t0 = time.time()
        if ablate or not self.stream:
            lg = super()._graph_logits(ctx, ablate)
            self._ph("decode", t0, 1, len(ctx))
            return lg
        d, s = self.dec, self._sctx
        if self._sbrain != self.brain.on:
            s, self._sbrain = None, self.brain.on
        new = self._new_tokens(s, ctx)
        if new is not None and d.pos + len(new) < d.max_len - 8:
            if self.graph_feed:                                # n79: graph segments on the static state, any length
                lg = d.feed_static(list(new))
            elif len(new) <= STEP_MAX:                         # a few tokens: graph steps
                for t in new:
                    lg = d.step(t)
            else:                                              # a pasted block: one exact eager feed onto the state
                d.unload_static(); lg = d.feed(list(new)); d.load_static(graphs=self.device.startswith("cuda"))
            kind, k = ("decode" if len(new) <= STEP_MAX else "prefill"), len(new)
        else:
            lg = self._read(ctx)
            kind, k = "prefill", len(ctx)
        self._sctx = list(ctx)
        lg = lg.float().clone()
        self._ph(kind, t0, k, d.pos)                           # context = everything the decoder has read
        return lg

    def _read(self, ctx):
        """A fresh prompt read. n79 (Glenn 10-25): CUDA-graph segments on the decoder's static buffers (feed_static), and
        a decide prompt starting with the constant tools block (184 of its ~199 tokens) restores the decoder state saved
        after that block (one per brain: a brain changes the weights) and reads only the rest: instant energy for the
        199-token read 45.0 -> 16.6 mJ / token (5080, logs/n79_prefill.log). DEMO_GRAPH_FEED=0 = the eager feed() read,
        DEMO_PREFIX_CACHE=0 = no state cache."""
        d, g = self.dec, self.device.startswith("cuda")
        d.reset()
        if not self.graph_feed:
            lg = d.feed(list(ctx)); d.load_static(graphs=g)
            return lg
        d.load_static(graphs=g)
        n = len(self._pfx_ids)
        if self.prefix_cache and n and len(ctx) > n and ctx[:n] == self._pfx_ids:
            c = self._pfx_cache.get(self.brain.on)
            if c is not None and c["sb"] == id(d.sb):
                d.restore_state(c); self.pfx_hits += 1
            else:
                d.feed_static(self._pfx_ids); self._pfx_cache[self.brain.on] = d.save_state()
            return d.feed_static(ctx[n:])
        return d.feed_static(list(ctx))

    @staticmethod
    def _new_tokens(s, ctx):
        """The tokens ctx adds to the stream the decoder holds (s = the context it last saw), or None (= start over).
        n78 10-25 (Glenn's raw-text test): the engine cuts every context to the model's block_size (2048); once a text
        reaches it, ctx SLIDES by one token per word and is no longer an extension of s, so each word re-read 2048
        tokens (8.8 kJ for one answer). The streaming decoder needs no cut (window 1024 + a fixed-size memory), so a
        slid ctx -- its start = the end of s -- continues the stream; the overlap must be >= 256 tokens or all of s."""
        if s is None or not ctx:
            return None
        n = len(s)
        if len(ctx) > n and ctx[:n] == s:                      # a plain extension
            return ctx[n:]
        for m in range(min(n, len(ctx) - 1), 255, -1):         # the longest overlap first: ctx[:m] == the last m of s
            if ctx[m - 1] == s[-1] and ctx[:m] == s[n - m:]:
                return ctx[m:]
        return None

    @torch.no_grad()
    def _traced_logits(self, ctx, ablate):
        t0 = time.time()
        lg = self._traced_impl(ctx, ablate)
        if self.v4f is not None:                               # n80: the v4f box, same tokens, same viz segment
            self._v4f_trace(ctx)
        self._ph("viz", t0, len(ctx), len(ctx))
        return lg

    def _traced_impl(self, ctx, ablate):
        """n78 10-25: the traced pass (brain view / meter) replayed from a CUDA graph of a right-padded forward (one per
        (length bucket, ablate)), as the parent does for untraced passes: an eager traced forward was ~170 ms, mostly
        host dispatch. The tracer's hooks only make GPU tensors, so they are captured with the graph; after a replay
        the tracer gets back the lists captured with it (same tensors, refilled) and Tv = the real length, so
        per_position reads only the real positions (causal model: the padding cannot reach them)."""
        if not (self.graph_trace and self.use_graph):
            return super()._traced_logits(ctx, ablate)
        T, tr = len(ctx), self.tracer
        L = min(int(self.c["block_size"]), -(-T // self.bucket) * self.bucket)
        key = (L, bool(ablate))
        if key not in self.tgraphs:
            static = torch.zeros(1, L, dtype=torch.long, device=self.device)

            def fw():
                tr.reset(L); tr.on = True
                try:
                    o = self.model(static, ablate_memory=ablate)
                finally:
                    tr.on = False
                return o[0] if isinstance(o, tuple) else o
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2):
                    fw()
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(g, pool=self.tpool):
                    out = fw()
            except Exception as e:                             # a host sync in a hook / the forward: eager for good
                print(f"[engine_stream] traced-pass graph capture failed at L={L}: {type(e).__name__}: {str(e)[:160]}; "
                      f"eager traced passes", flush=True)
                self.graph_trace = False
                return super()._traced_logits(ctx, ablate)
            self.tpool = g.pool()
            snap = dict(lins=list(tr.lins), acts=list(tr.acts), spikes=list(tr.spikes), coded=dict(tr.coded),
                        qv=list(tr.qv), audit=[b._audit for b in tr.spk_blocks])
            self.tgraphs[key] = (g, static, out, snap)
        g, static, out, snap = self.tgraphs[key]
        static.zero_(); static[0, :T] = torch.tensor(ctx, device=self.device)
        g.replay()
        tr.reset(L)
        tr.lins, tr.acts, tr.spikes, tr.coded, tr.qv = (list(snap["lins"]), list(snap["acts"]), list(snap["spikes"]),
                                                        dict(snap["coded"]), list(snap["qv"]))
        for b, au in zip(tr.spk_blocks, snap["audit"]):
            b._audit = au
        tr.Tv = T
        return out[0, T - 1].float().clone()

    def warmup(self):
        """Parent warm-up (memory-off graphs, dictionary, translator) + the decoder's two step graphs (HOLA cache off /
        on: the second needs a context past the first attn_chunk boundary)."""
        super().warmup()
        if self.device.startswith("cuda") and len(self.grammar.nodes) <= 20000:
            # every grammar node's mask on the device now: a first-visit copy waits behind the GPU queue (~2 ms each)
            for n in range(len(self.grammar.nodes)):
                self.table.mask(self.grammar, n, device=self.device)
        if self.device.startswith("cuda"):
            with self.lock:
                for T in (8, self.dec.C + 8):
                    self._sctx = None
                    self._graph_logits([1] * T, False)
                    self._graph_logits([1] * (T + 1), False)
                if self.graph_feed and self.dec.compile_read:
                    self.dec.warm_reads()                      # n81: every compiled segment now, none mid-answer
                if self.graph_feed:
                    # n79: the prompt-read segment graphs of the demo's usual prompt shapes (each captured once) and
                    # the tools-block state of the brain on now (other brains: saved on their first question)
                    for T in (143, 199, 330, 520, 760, 1100):
                        self._sctx = None
                        self._graph_logits([1] * T, False)
                    self._sctx = None
                    self._graph_logits(self._pfx_ids + self.encode("### Instruction:\nhello\n\n### Response:\n"), False)
                self._sctx = None
                if self.graph_trace and self.use_graph:        # traced-pass graphs of the two prompt buckets
                    for ab in (False, True):
                        for T in (8, self.bucket + 8):
                            self._traced_logits([1] * T, ab)

    def raw(self, text, emit, ablate=False, max_tokens=60, temperature=0.0, seed=0, remember=True):
        """Engine.raw with the history kept as TOKEN ids (n78 10-25): re-encoding the history text rarely reproduces
        the ids the model generated, so every new raw request broke the stream and restarted the decoder from the last
        2048 tokens. With the ids kept, a new request extends the stream exactly; the decoder reads on past the
        model's 2048 block (window 1024 + the fixed-size memory), and the meter's context is the true length."""
        hist = self.raw_history
        if hist and text[:1].isalnum():
            text = " " + text
        old = self.raw_ids if (hist and getattr(self, "raw_ids_for", None) == hist) else (self.encode(hist) if hist else [])
        ids = old + self.encode(text)
        bs = int(self.c["block_size"])
        if len(ids) > self.W:                                  # n78 10-25 (Glenn: keep bf16, disclose the limitation)
            emit(dict(type="note", msg=f"the text so far is {len(ids):,} tokens: attention reads the last {self.W:,}, "
                                       f"older text reaches the model only through its memory, which keeps the gist "
                                       f"rather than exact wording: copying text from further back is weaker, more so "
                                       f"in this demo's bf16 precision than in fp32 (the brain view shows the last "
                                       f"{bs:,} words). Reset to start over."))
        reply = self._decode(ids, len(self.encode(text)), max_tokens, emit, ablate, temperature, seed, phase="raw",
                             stop_eos=False)
        if remember:
            self.raw_history = hist + text + reply
            self.raw_ids, self.raw_ids_for = ids + self._last_out, self.raw_history
        emit(dict(type="done", mode="RAW", reply=reply))
        return reply

    def reset(self):
        super().reset()
        self._drop()
        self.raw_ids, self.raw_ids_for = [], None
