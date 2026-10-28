"""demo/server.py -- v2 2026-10-14 (Neuro 61; v1 2026-10-04 kept as server_v1_20261004.py). Web front end of the v4 demo;
the model side is scripts/demo/engine.py. One local user; stdlib HTTP + Server-Sent Events, no new dependencies.

  python scripts/demo/server.py --run sft_v4_reh13_v2tr --brain python=brain_py13r2 --brain health=brain_fdc13r3
  -> http://localhost:8765        (--device cpu for a slow test; --offline to start with the offline switch on)

Endpoints:
  GET  /, /info                       page, model info (incl. brains, tool protocol, energy meter assumptions)
  POST /ask      {text, ablate, offline, temperature, seed}   assistant turn (tool loop), SSE events:
                 turn | token (trace + energy) | tool_call | tool_result | done | error
  POST /raw      {text, ablate, max_tokens, temperature}       raw-text continuation (memory showcase)
  POST /replay   {}                    the last turn again with the memory layer switched off (history unchanged)
  POST /memtest  {N, trials, seed}     word pairs beyond the attention chunk, memory on vs off (memtest_progress | memtest)
  POST /brain    {name | "none" | "auto"}   one brain at a time ("auto" = picked per question); JSON {brain}
  POST /reset    {}                    forget the conversation (and start a new saved session)
  POST /resume   {name}                n81: continue a saved session (its history restored; new turns append to its file)
  GET  /sessions, /sessions/<file>    n80: the saved conversations (sessions.py: auto-saved after every request)"""
import os, sys, json, argparse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--brain", action="append", default=[], help="label=run (a run dir with adapter_step_*.pt)")
ap.add_argument("--default-brain", default="auto", help="brain at start: auto (router, n74 10-23 Glenn: the demo "
                "default) | none | a label")
ap.add_argument("--offline", action="store_true", help="start with every tool on its offline path")
ap.add_argument("--allow-mismatch", action="store_true", help="serve even if checkpoint and config disagree (debug)")
ap.add_argument("--budget", type=int, default=512, help="prompt token budget (the SFT length)")
ap.add_argument("--stream", type=int, default=1024, help="n78 (10-25 Glenn: the default): W > 0 = streaming decode + "
                "sliding attention window W (engine_stream.py; parity: logs/n78_engine_parity.log); 0 = the old "
                "full-forward engine (fp32)")
ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"], help="n78 (10-25 Glenn: bf16): streaming engine "
                "precision (bf16 weights + autocast = the trained regime)")
ARGS = ap.parse_args()

from engine import Engine, E_MAC, E_AC, E_SRAM, E_DRAM, PRECISION   # noqa: E402
import time                                                          # noqa: E402
from gpu_meter import GpuMeter, load_table, turn_report, nvml_index_of_cuda0   # noqa: E402  (n78 W4)

import sessions                                                     # noqa: E402  (n80)
ENG, SESSION = None, None
METER, TABLE = None, None   # n78 W4: measured GPU energy (gpu_meter.py) + the same-harness v4 / v4f reference
OFFLINE = [ARGS.offline]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def _page(self, b):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path.split("?", 1)[0] in ("/", "/index.html"):   # n81: "/?resume=<name>" is the page too
            self._page(open(os.path.join(HERE, "index.html"), "rb").read())
        elif self.path == "/info":
            self._json(dict(ENG.info, brain=ENG.brain.label, offline=OFFLINE[0],
                            energy=dict(mac_pj=E_MAC * 1e12, ac_pj=E_AC * 1e12, sram_read_pj=E_SRAM * 1e12,
                                        dram_read_pj=E_DRAM * 1e12, precision=PRECISION,
                                        source="45 nm, Horowitz ISSCC 2014"),
                            gpu=dict(name=METER.name if METER else None, table=TABLE),   # n78 W4
                            session=SESSION.name))                                        # n80
        elif self.path == "/sessions":                     # n80: the saved conversations
            self._page(sessions.list_page().encode())
        elif self.path.startswith("/sessions/"):
            f = self.path[len("/sessions/"):]
            if not sessions.NAME.match(f) or not os.path.exists(os.path.join(sessions.DIR, f)):
                return self._json({"error": "not found"}, 404)
            b = open(os.path.join(sessions.DIR, f), "rb").read()
            if f.endswith(".html"):
                self._page(b)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
                self.end_headers(); self.wfile.write(b)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
            assert isinstance(req, dict)
        except Exception:
            return self._json({"error": "bad request body (JSON object expected)"}, 400)
        if self.path == "/reset":
            with ENG.lock:
                ENG.reset(); SESSION.reset()
            return self._json({"ok": True})
        if self.path == "/resume":                         # n81: continue a saved session
            name = str(req.get("name", ""))
            if not sessions.NAME.match(name + ".json") or not os.path.exists(os.path.join(sessions.DIR, name + ".json")):
                return self._json({"error": "no such saved session"}, 404)
            with ENG.lock:
                ENG.reset()
                items, ENG.turns, ENG.raw_history = SESSION.resume(name)
            return self._json(dict(name=name, turns=len(ENG.turns), raw_chars=len(ENG.raw_history),
                                   items=[{k: it.get(k) for k in ("kind", "t", "text", "req", "events")} for it in items]))
        if self.path == "/brain":
            with ENG.lock:
                try:
                    b = ENG.brain.set(req.get("name"))
                    SESSION.note(f"brain: {b or 'none'}")
                    return self._json({"brain": b})
                except (KeyError, ValueError, FileNotFoundError) as e:
                    return self._json({"error": str(e), "brain": ENG.brain.label}, 400)
        if self.path == "/offline":
            OFFLINE[0] = bool(req.get("on"))
            return self._json({"offline": OFFLINE[0]})
        if self.path not in ("/ask", "/raw", "/replay", "/memtest"):
            return self._json({"error": "not found"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        evs = []                                           # n80: this request's events, for the saved session

        def emit(ev):
            evs.append(ev)
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode()); self.wfile.flush()

        with ENG.lock:
            temp, seed = float(req.get("temperature", 0)), int(req.get("seed", 0))
            off = bool(req.get("offline", OFFLINE[0]))
            if hasattr(ENG, "phases"):                     # n78 W4: timed GPU segments of this request
                ENG.phases = []
            t_req = time.time()
            try:
                if self.path == "/memtest":                # 10-20: one-click memory test (memory on vs off)
                    ENG.last = None                        # 10-21: it runs memory off itself; Replay replayed the turn before
                    ENG.memtest(emit, N=int(req.get("N", 8)), trials=int(req.get("trials", 20)), seed=seed)
                elif self.path == "/ask":
                    ENG.last = ("ask", req["text"], off, None)
                    ENG.assistant(req["text"], emit, ablate=bool(req.get("ablate")), offline=off, temperature=temp, seed=seed)
                elif self.path == "/raw":
                    ENG.last = ("raw", req["text"], int(req.get("max_tokens", 60)), ENG.raw_history)
                    ENG.raw(req["text"], emit, ablate=bool(req.get("ablate")), max_tokens=int(req.get("max_tokens", 60)),
                            temperature=temp, seed=seed)
                else:                                      # same turn again, memory off; history unchanged
                    if ENG.last is None:
                        emit(dict(type="error", msg="nothing to replay yet")); return
                    kind, text, x, hist_before = ENG.last
                    if kind == "ask":
                        saved = list(ENG.turns)
                        if saved and saved[-1][0] == text:     # the turn being replayed is not its own history
                            ENG.turns = saved[:-1]
                        try:
                            ENG.assistant(text, emit, ablate=True, offline=x, temperature=temp, seed=seed, remember=False)
                        finally:
                            ENG.turns = saved
                    else:
                        hist = ENG.raw_history
                        ENG.raw_history = hist_before
                        ENG.raw(text, emit, ablate=True, max_tokens=x, temperature=temp, seed=seed, remember=False)
                        ENG.raw_history = hist
                if METER is not None and self.path != "/memtest" and getattr(ENG, "phases", None):
                    emit(turn_report(METER, ENG.phases, t_req, time.time(), TABLE))   # n78 W4: measured GPU meter
            except (BrokenPipeError, ConnectionResetError):  # the page was closed / reloaded mid-answer: stop quietly
                return
            except Exception as e:                          # surface the failure in the page, keep serving
                try:
                    emit(dict(type="error", msg=f"{type(e).__name__}: {e}"))
                except OSError:
                    pass
                raise
            finally:                                        # n80: saved after every request (also a failed one)
                try:
                    SESSION.record(self.path, req, evs, t_req)
                except Exception as e:
                    print(f"[demo] session not saved: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    brains = dict(b.split("=", 1) for b in ARGS.brain)
    if ARGS.stream:
        from engine_stream import StreamEngine
        ENG = StreamEngine(ARGS.run, ARGS.device, window=ARGS.stream, dtype=ARGS.dtype, brains=brains,
                           allow_mismatch=ARGS.allow_mismatch, prompt_budget=ARGS.budget)
    else:
        ENG = Engine(ARGS.run, ARGS.device, brains=brains, allow_mismatch=ARGS.allow_mismatch, prompt_budget=ARGS.budget)
    if brains:
        ENG.brain.set(ARGS.default_brain)
    SESSION = sessions.Session(dict(ENG.info, brain=ENG.brain.label))   # n80
    if ARGS.device.startswith("cuda"):                 # n78 W4: the measured GPU meter (own-process NVML sampler)
        try:
            METER = GpuMeter(nvml_index_of_cuda0())
            TABLE = load_table(os.path.join(HERE, "gpu_energy_table.json"), METER.name)
            print(f"[demo] GPU meter: {METER.name}, v4 / v4f reference table "
                  f"{'loaded' if TABLE else 'MISSING for this GPU (run temp/n78_tables.sh)'}", flush=True)
        except Exception as e:
            print(f"[demo] GPU meter off: {type(e).__name__}: {e}", flush=True)
    import threading
    def _warm():                                                # n68: graphs, dictionary, translators off the first answers
        t = time.time(); ENG.warmup()
        print(f"[demo] warm-up done in {time.time() - t:.0f} s: answers and the GPU meter are at full speed now", flush=True)
    threading.Thread(target=_warm, daemon=True).start()
    i = ENG.info
    print(f"[demo] {i['run']} (step {i['step']}): {i['params']/1e6:.1f}M params, {i['attn_layers']} attention + "
          f"{i['spiking_layers']} memory layers, spike-coded {i['spike_coded']}, tool protocol v{i['tool_proto']}, "
          f"brains {i['brains']} (start: {ENG.brain.label or 'none'}) -> http://localhost:{ARGS.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler).serve_forever()
