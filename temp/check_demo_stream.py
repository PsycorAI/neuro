"""check_demo_stream.py -- 2026-10-04: summarize one /generate stream from scripts/demo/server.py (sanity of the
displayed rates and op counts vs the trainer's logged numbers)."""
import json, sys, urllib.request
text = sys.argv[1] if len(sys.argv) > 1 else "The history of the city begins in the year 1200, when a small group of"
req = urllib.request.Request("http://localhost:8765/generate", data=json.dumps(
    {"text": text, "max_tokens": 20}).encode(), headers={"Content-Type": "application/json"})
evs = []
for line in urllib.request.urlopen(req):
    line = line.decode().strip()
    if line.startswith("data: "):
        evs.append(json.loads(line[6:]))
toks = [e for e in evs if e["type"] == "token"]
mean = lambda xs: sum(xs) / len(xs)
h0 = mean([mean(e["h0"]) for e in toks])
print(f"{len(toks)} tokens | H0 spike rate {h0:.3f} | MLP rate per layer "
      f"{[round(mean([e['mlp'][l] for e in toks]), 4) for l in range(len(toks[0]['mlp']))]} | write {mean([e['write'] for e in toks]):.3f}")
print(f"active {mean([e['active'] for e in toks]):.4f} | ops event/dense (last) {toks[-1]['ops_event']/1e6:.1f}M / "
      f"{toks[-1]['ops_dense']/1e6:.1f}M = {toks[-1]['ops_dense']/toks[-1]['ops_event']:.2f}x | "
      f"pJ {toks[-1]['pj_event']:.0f} / {toks[-1]['pj_dense']:.0f}")
print("reply:", [e for e in evs if e["type"] == "done"][0]["reply"][:200])
