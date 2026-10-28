"""demo/server.py -- 2026-10-04 (Glenn OK). Stage 1 of docs/demo_brain_view_design.md: a local web page to chat with
a checkpoint while watching its neurons (spike raster), its memory write gate, an honest operation/energy meter, and
a memory-off replay switch. One local user; stdlib HTTP + Server-Sent Events, no new dependencies.

  python scripts/demo/server.py --run screen_r22a_x1_s0_30m --device cpu      # then open http://localhost:8765
  python scripts/demo/server.py --run sft_v3_363m --device cuda --chat

Nothing in src/ is modified: tracing uses forward hooks on nn.Linear / SparseAct, a wrapper around the spiking
blocks' forward_sequence (spike raster), and the existing `block.audit` capture (model.py, per-token write gate).

Operation meter (honest, per token, at the token's own position in the context):
  * every nn.Linear: dense cost = in*out MACs; event-driven cost = (nonzero inputs)*out. If the input is spikes
    (values in {-1,0,1}) the event-driven ops are accumulates (AC), else MACs.
  * attention: scores over the visible context (dense even with top-k) + value mix over top-k (or all) keys.
  * spiking-block memory state: read + erase + decay ~ 3*d_mem*d_head MACs.
  * energy: 45 nm model (Horowitz ISSCC 2014) MAC 4.6 pJ / AC 0.9 pJ, as src/energy.py. It ignores data movement
    (arXiv 2409.08290) and assumes event-driven hardware; a GPU computes the dense number.
  Not counted (small): embeddings / Engram lookups, norms, Canon taps, LIF updates."""
import os, sys, json, argparse, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--chat", action="store_true", help="default to the SFT chat template (else raw text)")
ARGS = ap.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F

if ARGS.device == "cpu":
    # The exact-LIF kernels are Triton (GPU only): swap in the torch reference, as temp/parity_r16_flags.py does.
    import lif_exact as LE

    def _lif_ml_ref(cur, beta, vth=1.0, K=None, alpha=2.0, gamma=1.0, levels=1, signed=False):
        B, T, N = cur.shape
        beta = beta if torch.is_tensor(beta) else torch.full((N,), float(beta))
        mem = torch.zeros(B, N, dtype=cur.dtype)
        outs = []
        for t in range(T):
            pre = beta * mem + cur[:, t]
            hard = torch.clamp(torch.floor(pre / vth), 0, levels)
            if signed:
                hard = hard - torch.clamp(torch.floor(-pre / vth), 0, levels)
            outs.append(hard)
            mem = pre - gamma * hard * vth
        return torch.stack(outs, 1), mem
    LE.lif_exact_ml = _lif_ml_ref
    LE.lif_exact = lambda cur, beta, vth=1.0, K=None, alpha=2.0, gamma=1.0: _lif_ml_ref(cur, beta, vth, gamma=gamma)

from chat import latest, build, load_tokenizer, PROMPT_FMT, STOP_MARKER   # noqa: E402
from hybrid_attn import SparseAct                                           # noqa: E402

E_MAC, E_AC = 4.6e-12, 0.9e-12
H0_BINS, MLP_BINS = 128, 48


class Tracer:
    """Collects per-position activity for one forward pass (batch 1)."""

    def __init__(self, model):
        self.model = model
        self.attn = [b for b in model.blocks if hasattr(b, "qkv")]
        self.spk_blocks = [b for b in model.blocks if not hasattr(b, "qkv")]
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.register_forward_hook(self._lin_hook)
            elif isinstance(m, SparseAct):
                m.register_forward_hook(self._act_hook)
        for b in self.spk_blocks:
            b.audit = True
            orig = b.forward_sequence

            def wrapped(*a, _orig=orig, **k):
                out = _orig(*a, **k)
                if self.on:
                    self.spikes.append(out[3][0].detach())          # (T, N) spike seq of this block
                return out
            b.forward_sequence = wrapped
        self.on = False
        self.reset(0)

    def reset(self, T):
        self.T, self.lins, self.acts, self.spikes = T, [], [], []
        for b in self.spk_blocks:
            b._audit = None                     # the memory-off path skips the capture; don't show a stale one

    def _lin_hook(self, mod, inp, out):
        if not self.on:
            return
        x = inp[0]
        if x.dim() != 3 or x.shape[0] != 1 or x.shape[1] != self.T:
            return                                                  # not a per-position projection (batch 1)
        x = x[0].detach()
        nz = x != 0
        binary = bool(((x == 0) | (x.abs() == 1)).all())
        self.lins.append((nz.sum(-1).float(), mod.in_features, mod.out_features, binary))

    def _act_hook(self, mod, inp, out):
        if self.on and out.dim() == 3 and out.shape[1] == self.T:
            self.acts.append((out[0] != 0).detach())                # (T, hidden) fire mask

    def per_position(self, t0):
        """Trace dicts for positions t0..T-1."""
        T = self.T
        ev_mac = torch.zeros(T); ev_ac = torch.zeros(T); dense = torch.zeros(T)
        for nnz, i, o, binary in self.lins:
            nnz = nnz.cpu()
            dense += i * o
            (ev_ac if binary else ev_mac).add_(nnz * o)
        ctx = torch.arange(1, T + 1, dtype=torch.float32)
        for b in self.attn:
            dm = b.n_heads * b.d_head
            w = getattr(b, "attn_window", 0) or 0
            vis = ctx.clamp(max=w) if w > 0 else ctx
            mix = vis.clamp(max=b.attn_topk) if b.attn_topk else vis
            ev_mac += dm * vis + dm * mix
            dense += 2 * dm * vis
        for b in self.spk_blocks:
            mem_ops = 3 * b.d_mem * b.d_head
            ev_mac += mem_ops; dense += mem_ops
        spk = [s.ne(0).float().cpu() for s in self.spikes]
        acts = [a.float().cpu() for a in self.acts]
        writes = [b._audit["w"][0].float().mean(-1).cpu() for b in self.spk_blocks
                  if getattr(b, "_audit", None) is not None]
        n_units = sum(s.shape[1] for s in spk) + sum(a.shape[1] for a in acts)
        res = []
        for t in range(t0, T):
            fired = sum(s[t].sum() for s in spk) + sum(a[t].sum() for a in acts)
            h0 = (F.adaptive_avg_pool1d(spk[0][t][None, None], H0_BINS)[0, 0].tolist() if spk else [])
            res.append(dict(
                h0=[round(v, 3) for v in h0],
                mlp=[round(float(a[t].mean()), 4) for a in acts],
                mlp_strips=[[round(v, 3) for v in F.adaptive_avg_pool1d(a[t][None, None], MLP_BINS)[0, 0].tolist()]
                            for a in acts],
                write=round(float(writes[0][t]), 3) if writes else None,
                active=round(float(fired) / max(n_units, 1), 4),
                ops_event=float(ev_mac[t] + ev_ac[t]), ops_dense=float(dense[t]),
                pj_event=float(ev_mac[t] * E_MAC + ev_ac[t] * E_AC) * 1e12, pj_dense=float(dense[t] * E_MAC) * 1e12))
        return res


class Demo:
    def __init__(self):
        blob = torch.load(latest(ARGS.run), map_location=ARGS.device, weights_only=False)
        self.c = c = blob["cfg"]
        self.model = build(c, ARGS.device)
        self.model.load_state_dict(blob["model"], strict=False)
        self.model.eval()
        if c.get("lut_subword_fallback", False):
            import lut_spell
            self.encode = lut_spell.make_encoder(c["vocab"], subword_fallback=True)
            self.decode = lut_spell.make_decoder(c["vocab"])
        else:
            self.encode, self.decode = load_tokenizer(c["vocab"])
        self.tracer = Tracer(self.model)
        self.lock = threading.Lock()
        self.history = ""                       # all text so far (context = its last block_size tokens)
        self.last = None                        # (history_before, user_text, chat) for the memory-off replay
        tr = self.tracer
        self.info = dict(run=ARGS.run, device=ARGS.device, tokens_trained=blob.get("tokens", 0),
                         params=sum(p.numel() for p in self.model.parameters()),
                         n_layers=len(self.model.blocks), attn_layers=len(tr.attn), spiking_layers=len(tr.spk_blocks),
                         n_neurons=c.get("n_neurons"), block_size=c["block_size"],
                         attn_window=c.get("attn_window_fixed", 0) or 0, attn_topk=c.get("attn_topk", 0),
                         chat_default=ARGS.chat, h0_bins=H0_BINS)

    @torch.no_grad()
    def run(self, history, user, chat, max_tokens, temperature, ablate, seed, emit, attn_window=None):
        # Inference attention window (0 = full context): with a short window, anything older must come through the
        # spiking memory. Default = what the checkpoint was trained with.
        w = self.info["attn_window"] if attn_window is None else int(attn_window)
        for b in self.tracer.attn:
            b.attn_window = w
        new_text = PROMPT_FMT.format(message=user) if chat else user
        hist_ids = self.encode(history) if history else []
        new_ids = self.encode(new_text)
        ids = hist_ids + new_ids
        stop_ids = self.encode(STOP_MARKER) if chat else None
        gen = torch.Generator().manual_seed(seed)
        bs = self.c["block_size"]
        out_ids, n_prompt = list(ids), len(new_ids)
        for step in range(max_tokens + 1):
            ctx = out_ids[-bs:]
            T = len(ctx)
            self.tracer.reset(T); self.tracer.on = True
            logits = self.model(torch.tensor([ctx], device=ARGS.device), ablate_memory=ablate)
            self.tracer.on = False
            logits = (logits[0] if isinstance(logits, tuple) else logits)[0, -1].float().cpu()
            if step == 0:                       # this turn's prompt tokens (those still inside the window)
                k = min(n_prompt, T)
                for tid, tr in zip(ctx[T - k:], self.tracer.per_position(T - k)):
                    emit(dict(type="token", kind="prompt", text=self.decode([tid]), **tr))
            else:
                emit(dict(type="token", kind="gen", text=self.decode([out_ids[-1]]),
                          **self.tracer.per_position(T - 1)[0]))
            if step == max_tokens:
                break
            if not self.c.get("lut_subword_fallback", False):
                logits[0] = float("-inf")       # id 0 = out-of-LUT token in pre-fallback corpora; decodes to <unk>
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                p = F.softmax(logits / temperature, -1)
                nxt = int(torch.multinomial(p, 1, generator=gen))
            out_ids.append(nxt)
            if stop_ids and out_ids[-len(stop_ids):] == stop_ids:
                break
        reply = self.decode(out_ids[len(ids):])
        if stop_ids and reply.endswith(STOP_MARKER):
            reply = reply[: -len(STOP_MARKER)]
        return new_text, reply


DEMO = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            b = open(os.path.join(HERE, "index.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        elif self.path == "/info":
            self._json(DEMO.info)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/reset":
            with DEMO.lock:
                DEMO.history, DEMO.last = "", None
            return self._json({"ok": True})
        if self.path not in ("/generate", "/replay"):
            return self._json({"error": "not found"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(ev):
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode()); self.wfile.flush()

        with DEMO.lock:
            temp, maxt = float(req.get("temperature", 0)), int(req.get("max_tokens", 60))
            seed = int(req.get("seed", 0))
            win = req.get("attn_window")
            try:
                if self.path == "/generate":
                    chat = bool(req.get("chat", DEMO.info["chat_default"]))
                    hist = DEMO.history
                    new_text, reply = DEMO.run(hist, req["text"], chat, maxt, temp, bool(req.get("ablate")), seed, emit,
                                               win)
                    DEMO.last = (hist, req["text"], chat)
                    DEMO.history = hist + new_text + reply + ("\n\n" if chat else "")
                else:                                   # same turn again, memory on/off; history unchanged
                    if DEMO.last is None:
                        emit(dict(type="error", msg="nothing to replay yet")); return
                    hist, text, chat = DEMO.last
                    _, reply = DEMO.run(hist, text, chat, maxt, temp, bool(req.get("ablate", True)), seed, emit, win)
                emit(dict(type="done", reply=reply))
            except Exception as e:                      # surface the failure in the page, keep serving
                emit(dict(type="error", msg=f"{type(e).__name__}: {e}"))
                raise


if __name__ == "__main__":
    DEMO = Demo()
    i = DEMO.info
    print(f"[demo] {i['run']}: {i['params']/1e6:.1f}M params, {i['spiking_layers']} spiking + {i['attn_layers']} "
          f"attention layers, context {i['block_size']} -> http://localhost:{ARGS.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler).serve_forever()
