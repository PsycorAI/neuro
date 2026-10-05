# Demo web page: live brain view + energy meter — design (2026-10-04)

Status: **design only, no code yet.** Audience: Glenn's son, Oct 19. Glenn's spec (10-04): a simple web page with a
live brain view and an energy meter; other visuals that convey our advantages are welcome.

## 1. What it is

A local web page (one Python server + one static HTML/JS page) where you chat with the model, and while it reads and
writes text you see what its neurons and memory are doing, token by token. It is a **demo instrument**, not a product
UI and not a general interpretability toolkit. Every number it shows must be real and checkable: a demo that
overstates (energy, memory) would hurt more than help once the son looks closely.

Panels (priority order):
1. **Chat** with streamed tokens.
2. **Spike raster**: per generated/read token, which neurons fired (H0 spiking layer + the 5%-active MLP layers),
   drawn like an EEG strip. Shows "only ~5% of neurons are active".
3. **Memory-off switch**: re-run the last answer with the memory ablated (`ablate_memory=True`, already in
   `model.forward`). If the answer depended on the memory, it visibly forgets.
4. **Energy meter**: operations per token for our model vs a same-size dense transformer, with the assumptions shown.
5. **Memory write/read view + "remembered from N tokens ago"**: when a fact is told, which tokens wrote strongly into
   memory; when it is asked later, which earlier tokens the memory read is attributed to, and how far back they are
   vs the attention window.
6. **Neuron inspector**: click a neuron → the tokens/contexts that activate it most (precomputed offline).
7. **Tool loop**: the model emits `TOOL {...}`; the server runs it (search/calculator) and feeds the result back.

NOT in scope: training changes, a hosted service, multi-user, mobile layout polish.

## 2. Why now vs defer

Build now, staged. The UI and server do not need the GPUs (CPU or the 5080 with an existing checkpoint), so they
run in parallel with the r21/r24 memory screens and the v4 pretrain. Waiting until v4 finishes (~Oct 14) would leave
too little time.

**But panels 3 and 5 are gated on the v4 memory actually being used** (r21c: memory Δ 0.00 at 12 layers; r21g:
partial). If v4 ships with attention-only recall, the memory-off switch would show nothing, and we must not demo it.
Panels 1, 2, 4, 6 work regardless.

## 3. Where it plugs into the code

Existing pieces (verified 10-04):
- `scripts/chat.py:79` `generate_response` — prompt template (`### Instruction:` / `### Response:`), stop on `###`,
  sampling. Recomputes the full context every token (`chat.py:93`); no KV cache is used. Persistent brain state across
  turns via `initial_state` / `return_final_state` (`chat.py:158-168`; `model.py:1429`, `model.py:1736` save_brain).
- `src/model.py:1429` `forward(..., ablate_memory=, return_stats=, initial_state=, return_final_state=, kv_cache=)`.
  `kv_cache` incremental decode exists (FPT single-pass only) but chat.py doesn't use it.
- `src/model.py:704` `forward_sequence` returns `spk_seq (B,T,N)` for the spiking block (collected only under
  `return_stats`, `model.py:1611`). Write-gate telemetry is aggregate only (`_last_beta_stats`, `model.py:842-999`);
  **per-token β, k, q are not exposed** anywhere.
- `src/hybrid_attn.py:137` `SparseAct.forward`: `fire = a > theta` — a forward hook can capture the per-token fire
  mask without code changes.
- `src/energy.py:45` `spiking_energy_per_token`: **stale for v4** — counts the attention-block FFN as dense
  `8 d^2` MAC and knows nothing of the 5% relu2 MLP or top-k attention.
- `src/tool_data.py` protocol: `TOOL <json>` / `ANSWER <text>`; 9 tools in `training_data/posttrain/tools.json`
  (search, calculator, unit_convert, get_weather, get_time, define, translate, set_reminder, send_message).
  The SFT data is **single-step** (prompt → call). There is no "tool result → final answer" turn format and no
  executor anywhere.

New code:
- `scripts/demo_server.py` — stdlib `http.server.ThreadingHTTPServer` + Server-Sent Events (no new dependencies;
  single local user). Endpoints: `POST /chat` (SSE stream of `{token, trace}` events), `POST /replay?ablate=1`,
  `GET /neuron/<layer>/<id>`, `POST /tool`. Loads any checkpoint through `train_gpu.build_model` (same as chat.py).
- `scripts/demo/index.html` — vanilla JS + `<canvas>` (raster/heatmap), no build step.
- `scripts/demo_neuron_index.py` — offline: run ~1M val tokens, keep top-20 contexts per neuron → JSON.
- **One small additive change to `model.py`** (needs Glenn's OK; default off = byte-identical; smoke-test before any
  queued run starts): a `self._trace` dict that, when set by the demo, stores per-token `beta_seq`, `k`, `q` and the
  decay gate of the H0 block. Hooks alone can't reach these tensors (they are locals inside `forward_sequence`).
- `src/energy.py`: a v4-aware variant (see §4 / §5) — new function, old one untouched.

Dev checkpoints: `screen_r22a_x1_s0_30m` (v4 family: H0 + attention + 5% relu2 MLP + x1; memory works, haystack
100%) for the brain view; `sft_v3_363m` for chat + tool format. Swap to v4 by run name.

## 4. Cost & risk

- Compute: tracing adds per-token tensor copies; at 1B on the 5090 the full-recompute decode is the bottleneck
  (512-token context). Target ≥ 10 tok/s with tracing; if slower, use `kv_cache` or trace only every token's last
  position.
- **Risk 1 — overstated energy.** The honest v4 number is modest. Per attention layer at d=2048, h=12288:
  qkv+o 4d² = 16.8M MAC, MLP up-projection d·h = 25.2M (dense, the threshold needs every pre-activation), MLP down
  25.2M × 5% = 1.3M → ~43M vs 67M dense ≈ **1.5× fewer ops**, not 10×, and only on event-driven hardware. Data
  movement is ignored by the standard 45 nm MAC/AC model (arXiv 2409.08290, 2309.03388). The meter must say so.
  Showing "8–10× smaller circuits" (the r19 interpretability result) is a different, defensible claim — keep the two
  separate.
- **Risk 2 — attribution that isn't causal.** A delta-rule memory has no discrete slots; "which earlier tokens did
  this read come from" is an approximation (ignores the erase term). It must be checked by knockout (§5) or not shown.
- **Risk 3 — memory switch shows nothing** if v4's memory is unused (see §2 gate).
- Risk 4 — 512-token context: multi-turn chat + tool results overflow fast. Carried brain state helps only if the
  model was trained to use carried state (r24b stateful).
- Risk 5 — tool loop needs a result-conditioned answer format the model was never trained on → needs ~2–5k SFT
  examples (`### Tool result:` turn) in the demo SFT stage.

## 5. Implementation specifics

**Memory attribution** (H0 GDN-2 delta memory, per head): the read at step t is `o_t = M_t q_t`, with
`M_t = a_t M_{t-1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`. Approximate source weight of token s for the read at t:
`w_{t,s} = β_s · Π_{u=s+1..t} a_u · (k_s · q_t)` (erase term dropped). Show the top-3 sources and `t − s` ("N tokens
ago"), next to the attention window size. **Knockout check**: zero `β_s` for the top source and re-run; show the
attribution only if the answer's log-prob drops (pre-registered below).

**Energy meter** (`energy.py` v4 variant): per token, count MACs for qkv/o, top-k attention scores (still dense
over the window), value mix over k=16 keys, MLP up-projection (dense), MLP down-projection × measured fire rate, H0
spike-driven projections as ACs × measured spike rate, LM head; same-size dense transformer as baseline. Display
both totals, the ratio, and a one-line assumptions note.

**Spike raster**: per token, H0 `spk_seq` row (N=8192 at 1B → downsample to a fixed 256-pixel strip by max-pool,
labelled) + per-layer MLP fire fraction as a small bar.

**Tool loop**: decode until `TOOL {json}` parses (`tool_data.parse_response`), run the tool (search = Wikipedia REST
summary with an offline cached fallback for the rehearsed questions, so the demo can't fail on network), append
`### Tool result:\n<text>\n\n### Response:\n` and continue. "Search when unsure": SFT examples first; an entropy
threshold on the first answer tokens as a fallback trigger. The hippocampal familiarity signal is a research idea
(to-do 10-04), not a demo dependency.

## 6. Validation (pre-registered)

The project has a history of results that look real and don't survive a re-check; for a demo the equivalent failure is
a visual that implies something false. Criteria, on scripted cases built from the haystack eval (fact at gap 64/192/440):
- **Memory-off switch**: shown only if, on 50 scripted facts beyond the attention window, recall is ≥ 80% with memory
  and ≤ chance + 10 pt with memory off. (30M dev model passes today: haystack 100%; v4 unknown.)
- **Attribution**: shown only if knocking out the top-attributed source drops the correct answer's log-prob by
  ≥ 1 nat in ≥ 80% of the 50 cases; otherwise show write/read activity only, without "from N tokens ago".
- **Energy meter**: op counts unit-tested against hand calculation for the 30M and 1B shapes; assumptions on screen.
- **Spike raster**: displayed rates equal the trainer's `[spike]` / MLP `last_rate` numbers on the same text (±1%).
- **Reliability**: 10 rehearsed scenarios end-to-end, 3 times each, no failure; ≥ 10 tok/s on the 5090.
- Baseline for "is this impressive": the same page pointed at the G3 transformer (`g3_tfmr_150m`) — no raster, no
  memory, dense meter. Showing both side by side is the honest comparison.

## 7. Recommendation & decision gate

**Build now, in stages, on CPU / existing checkpoints:**
- S1 (≈2 days): server + chat + spike raster + energy meter (v4-aware, honest) + memory-off switch.
- S2 (after r24 results, ~Oct 6): memory write/read view + attribution + knockout check. Gate: v4 memory used
  (memory Δ > 0 and beyond-window haystack above chance). If not, drop S2 and the switch from the v4 demo.
- S3: neuron inspector (offline index on v4 once a mid-run checkpoint exists).
- S4: tool loop + `### Tool result:` SFT data (fold into the demo SFT stage).
Needs Glenn's OK for the small default-off `model.py` trace flag.

## References

1. Port target for the energy honesty: *Reconsidering the Energy Efficiency of SNN Inference from Analytical
   Perspectives* (arXiv 2409.08290) and *Are SNNs Truly Energy-efficient? A Hardware Perspective* (arXiv 2309.03388)
   — MAC/AC counting ignores data movement; gains need high sparsity on event-driven hardware.
2. UI patterns: VisuaLLM (INLG 2023 demos; local generation backend + web front end), Synapse (live activation
   stats over WebSockets), Full GPT-2 Visualizer (HF Space: neuron explorer, activation patching).
3. Energy model in use: Horowitz ISSCC 2014 45 nm (MAC 4.6 pJ, AC 0.9 pJ), as in `src/energy.py`.
4. Memory mechanism: GDN-2 / gated delta rule (`src/model.py` H0 block); SWAX arXiv 2509.24552 for the
   attention-window context.
