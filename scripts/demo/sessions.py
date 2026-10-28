"""demo/sessions.py -- 2026-10-26 (Neuro 80, Glenn's overnight item 3): every demo conversation saved to a file and linked
from the page. The server hands each request's events to Session.record; after every request the session is rewritten
as logs/demo_sessions/session_<YYYYMMDD_HHMMSS>.json (compact: the conversation, tool calls / results, the GPU meter's
summary, the generated words with their activity; no prompt traces) + a self-contained .html view of the same. Reset
starts a new session file. GET /sessions = the list page, GET /sessions/<file> = one file (strict name: no paths).
n81: "continue" (/?resume=<name> -> POST /resume) loads a saved session back into the demo; new turns append to its file."""
import os, re, json, time, html

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(ROOT, "logs", "demo_sessions")
NAME = re.compile(r"^session_\d{8}_\d{6}\.(json|html)$")
KEEP_GPU = ("model_J", "decode_tokens", "mJ_tok", "tok_s", "first_token_s", "idle_W", "context")


class Session:
    def __init__(self, info):
        self.info, self.items, self.name = info, [], None

    def reset(self):
        self.items, self.name = [], None

    def note(self, text):
        """A line without a request (brain switch, offline switch)."""
        if self.items:
            self.items.append(dict(kind="note", t=time.time(), text=text))
            self.save()

    def record(self, path, req, events, t0):
        """One request: its path (/ask, /raw, /replay, /memtest), the request fields and the events it emitted."""
        if self.name is None:
            self.name = time.strftime("session_%Y%m%d_%H%M%S", time.localtime(t0))
        it = dict(kind=path.strip("/"), t=t0, secs=round(time.time() - t0, 2),
                  req={k: req[k] for k in ("text", "ablate", "offline", "temperature", "max_tokens") if k in req},
                  prompt_words=0, words=[], events=[])
        for ev in events:
            ty = ev.get("type")
            if ty == "token" and ev.get("kind") == "prompt":
                it["prompt_words"] += 1
            elif ty == "token":                                    # a generated word with its activity
                w = dict(text=ev.get("text", ""), phase=ev.get("phase"), active=ev.get("active"))
                if "active_f" in ev:
                    w["active_f"] = ev["active_f"]
                it["words"].append(w)
            elif ty == "gpu_energy":
                g = {k: ev.get(k) for k in KEEP_GPU}
                g["viz_J"] = (ev.get("by") or {}).get("viz", {}).get("J")
                if ev.get("ref"):
                    g["ref_v4_mJ"], g["ref_v4f_mJ"] = ev["ref"]["v4"]["mJ_tok"], ev["ref"]["v4f"]["mJ_tok"]
                it["gpu"] = g
            elif ty in ("text", "memtest_progress"):
                continue
            else:                                                  # turn, tool_call, tool_result, done, note, error, memtest
                it["events"].append(ev)
        self.items.append(it)
        self.save()

    def resume(self, name):
        """n81 (Glenn 10-26: continue a saved session; new turns APPEND to the same file): this session becomes the saved
        one <name>, after a "resumed at" note. Returns (items, turns, raw_history): the engine history rebuilt as the
        engine built it (see history())."""
        doc = json.load(open(os.path.join(DIR, name + ".json")))
        self.items, self.name = doc["items"], name             # the file's name (validated), not the name stored in it
        turns, raw = history(self.items)
        self.items.append(dict(kind="note", t=time.time(), text=time.strftime("resumed at %Y-%m-%d %H:%M:%S (earlier turns: text only)")))
        self.save()
        return self.items, turns, raw

    def save(self):
        os.makedirs(DIR, exist_ok=True)
        base = os.path.join(DIR, self.name)
        doc = dict(name=self.name, model=self.info, items=self.items)
        with open(base + ".json.tmp", "w") as f:
            json.dump(doc, f)
        os.replace(base + ".json.tmp", base + ".json")
        with open(base + ".html.tmp", "w", encoding="utf-8") as f:
            f.write(render(doc))
        os.replace(base + ".html.tmp", base + ".html")


def _reply(it):
    for ev in it["events"]:
        if ev.get("type") == "done":
            return ev.get("reply") or ""
    if it["kind"] == "raw":
        return "".join(w["text"] for w in it["words"] if w.get("phase") == "raw")
    return ""


def history(items):
    """n81: the engine's conversation state after these saved requests, as Engine.assistant / raw left it: every finished
    /ask turn -> (question, reply) (canned backend replies included, as the engine keeps them); every finished /raw ->
    the raw text (with the engine's leading-space rule); replays and memory tests never entered it; a request without a
    "done" event (failed) did not either."""
    turns, raw = [], ""
    for it in items:
        done = next((ev for ev in it.get("events", []) if ev.get("type") == "done"), None)
        if done is None:
            continue
        if it["kind"] == "ask":
            turns.append((it["req"].get("text", ""), done.get("reply") or ""))
        elif it["kind"] == "raw":
            t = it["req"].get("text", "")
            if raw and t[:1].isalnum():
                t = " " + t
            raw += t + (done.get("reply") or "")
    return turns, raw


def render(doc):
    """A self-contained page of one session (no scripts, no external files)."""
    e = lambda s: html.escape(str(s))
    m = doc.get("model") or {}
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
           f" initial-scale=1'><title>{e(doc['name'])}</title><style>{CSS}</style></head><body><main>"
           f"<h1>Demo session {e(doc['name'][8:].replace('_', ' '))}</h1>"
           f"<p class='sub'>model {e(m.get('run', '?'))} &middot; {e(m.get('dtype', ''))} &middot; attention window "
           f"{e(m.get('attn_window', ''))} &middot; <a href='/sessions'>all sessions</a> &middot; "
           f"<a href='/sessions/{e(doc['name'])}.json'>JSON</a> &middot; "
           f"<a href='/?resume={e(doc['name'])}'>continue this session</a></p>"]
    for it in doc["items"]:
        when = time.strftime("%H:%M:%S", time.localtime(it["t"]))
        if it["kind"] == "note":
            out.append(f"<div class='note'>{when} &middot; {e(it['text'])}</div>"); continue
        q = it["req"].get("text")
        label = {"ask": "you", "raw": "you (raw text)", "replay": "replay, memory OFF", "memtest": "memory test"}[it["kind"]]
        if q:
            out.append(f"<div class='msg user'><span class='tag'>{when} &middot; {label}</span>{e(q)}</div>")
        elif it["kind"] != "ask":
            out.append(f"<div class='note'>{when} &middot; {label}</div>")
        body, tag = [], "assistant" if it["kind"] == "ask" else ("model" if it["kind"] == "raw" else it["kind"])
        if it["req"].get("ablate") or it["kind"] == "replay":
            tag += " (memory OFF)"
        for ev in it["events"]:
            ty = ev.get("type")
            if ty == "turn" and ev.get("auto"):
                tag += f" &middot; router: {e(ev.get('brain') or 'no brain')}"
            elif ty == "tool_call":
                body.append(f"<div class='card'><i>{e(ev.get('say', ''))}</i><br><b>{e(ev.get('name'))}</b>"
                            f"({e(json.dumps(ev.get('arguments')))})</div>")
            elif ty == "tool_result":
                body.append(f"<div class='card'>result <span class='src'>{e(ev.get('source', ''))}</span>"
                            f"{' <b class=bad>failed</b>' if ev.get('ok') is False else ''}<br>{e(ev.get('text', ''))}</div>")
            elif ty == "tool_dropped":
                body.append(f"<div class='fix'>backend: dropped the call {e(ev.get('name'))} (unrelated to the message)</div>")
            elif ty == "note":
                body.append(f"<div class='fix'>backend: {e(ev.get('msg'))}</div>")
            elif ty == "error":
                body.append(f"<div class='fix bad'>error: {e(ev.get('msg'))}</div>")
            elif ty == "memtest":
                body.append(f"<div>memory ON {ev.get('on')}/{ev.get('trials')} correct, memory OFF {ev.get('off')}/"
                            f"{ev.get('trials')} (guessing {ev.get('chance')}%)</div>")
        rep = _reply(it)
        if rep:
            body.append(f"<div>{e(rep)}</div>")
        ws = [w for w in it["words"] if w.get("active") is not None]
        stats = []
        if ws:
            stats.append(f"{len(ws)} words written, v4 neurons active {100 * sum(w['active'] for w in ws) / len(ws):.1f}%")
            wf = [w["active_f"] for w in ws if "active_f" in w]
            if wf:
                stats.append(f"v4f {100 * sum(wf) / len(wf):.1f}%")
        g = it.get("gpu")
        if g and g.get("model_J") is not None:
            stats.append(f"GPU {g['model_J']:.1f} J ({g.get('mJ_tok') or 0:.0f} mJ / word, {g.get('tok_s') or 0:.0f} words/s)")
            if g.get("ref_v4f_mJ"):
                stats.append(f"table at {g.get('context')} words: v4 {g['ref_v4_mJ']:.0f} vs v4f {g['ref_v4f_mJ']:.0f} mJ / word")
        stats.append(f"{it['secs']:.1f} s")
        out.append(f"<div class='msg bot'><span class='tag'>{tag}</span>{''.join(body)}"
                   f"<div class='sub'>{' &middot; '.join(stats)}</div></div>")
    out.append("</main></body></html>")
    return "".join(out)


def list_page():
    os.makedirs(DIR, exist_ok=True)
    rows = []
    for f in sorted(os.listdir(DIR), reverse=True):
        if not (NAME.match(f) and f.endswith(".json")):
            continue
        try:
            d = json.load(open(os.path.join(DIR, f)))
        except (OSError, ValueError):
            continue
        n = sum(1 for it in d["items"] if it["kind"] != "note")
        first = next((it["req"].get("text") for it in d["items"] if it.get("req", {}).get("text")), "")
        stem = f[:-5]
        rows.append(f"<tr><td><a href='/sessions/{stem}.html'>{html.escape(stem[8:].replace('_', ' '))}</a></td><td>{n}</td>"
                    f"<td>{html.escape(first[:90])}</td><td><a href='/sessions/{stem}.json'>JSON</a> &middot; "
                    f"<a href='/?resume={stem}'>continue</a></td></tr>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            f" initial-scale=1'><title>Demo sessions</title><style>{CSS}</style></head><body><main><h1>Demo sessions</h1>"
            f"<p class='sub'>Saved automatically after every answer (logs/demo_sessions); Reset starts a new one. "
            f"<a href='/'>back to the demo</a></p><table><tr><th>started</th><th>requests</th><th>first message</th><th></th></tr>"
            + "".join(rows) + "</table></main></body></html>")


CSS = ("body{margin:0;background:#0b0f17;color:#e6ebf5;font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}"
       "main{max-width:860px;margin:0 auto;padding:16px}h1{font-size:18px}a{color:#5be0ff}"
       ".sub{color:#8a96ad;font-size:12px;margin-top:4px}.msg{padding:8px 12px;border-radius:10px;margin:10px 0;"
       "white-space:pre-wrap;word-break:break-word}.user{background:#1b2740;margin-left:10%}"
       ".bot{background:#161f30;border:1px solid #1f2a3d}.tag{font-size:11px;color:#8a96ad;display:block;margin-bottom:2px}"
       ".card{margin:6px 0;border:1px solid #2f4266;border-radius:8px;padding:6px 10px;background:#0e1420;font-size:13px}"
       ".card b{color:#5be0ff}.src{font-size:11px;border:1px solid #1f2a3d;border-radius:999px;padding:1px 8px;color:#8a96ad}"
       ".fix{font-size:11px;color:#ffb347;margin:3px 0}.bad{color:#ff6b6b}.note{color:#8a96ad;font-size:12px;margin:8px 0}"
       "table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #1f2a3d;padding:6px;text-align:left;"
       "vertical-align:top}th{color:#8a96ad;font-weight:500}")
