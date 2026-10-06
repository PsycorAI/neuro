import json
for l in open("/home/glenn/projects/neuro/logs/scorecard.jsonl"):
    d = json.loads(l)
    c = d.get("circuit") or d.get("circuits") or {}
    print(d.get("name"), (c.get("lm") or {}).get("entries90_med"), (c.get("mqar") or {}).get("entries90_med"))
