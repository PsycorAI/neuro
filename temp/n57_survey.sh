#!/bin/bash
# Neuro 57: read-only survey for the demo plan (data samples, cached datasets / models, installed packages).
cd /home/glenn/projects/neuro || exit 1
PY=/home/glenn/projects/bdh/venv/bin/python
echo "== tools.jsonl: one record per kind + tool counts"
$PY - <<'EOF'
import json, collections
kinds, names, seen = collections.Counter(), collections.Counter(), {}
for line in open("training_data/posttrain/tools.jsonl", encoding="utf-8"):
    r = json.loads(line); kinds[r.get("kind")] += 1
    if r["target_mode"] == "TOOL":
        n = r["target_call"]["name"]; names[n] += 1
        if n not in seen: seen[n] = r
print(dict(kinds)); print(dict(names))
for n, r in seen.items():
    print(" ", n, "|", r["instruction"][:90].replace("\n", " "), "->", json.dumps(r["target_call"]["arguments"])[:110])
EOF
echo "== abstain.jsonl: unanswerable samples + counts"
$PY - <<'EOF'
import json, collections
c, shown = collections.Counter(), 0
for line in open("training_data/posttrain/abstain.jsonl", encoding="utf-8"):
    r = json.loads(line); c[r["pair_type"]] += 1
    if r["pair_type"] == "unanswerable" and shown < 5:
        shown += 1; print(" ", r["prompt"][:110], "->", r["chosen"][:60])
print(dict(c))
EOF
echo "== HF cache: datasets and models"
ls ~/.cache/huggingface/datasets 2>/dev/null | head -n 30
ls ~/.cache/huggingface/hub 2>/dev/null | head -n 40
echo "== packages"
$PY -m pip list 2>/dev/null | grep -i -E "^(pint|nltk|python-dateutil|rank-bm25|sentencepiece|sacremoses|transformers|torch|datasets|peft|requests|tzdata|ctranslate2|argostranslate) "
$PY -c "import sqlite3; c=sqlite3.connect(':memory:'); c.execute('create virtual table t using fts5(x)'); print('sqlite fts5 ok', sqlite3.sqlite_version)"
echo "== disk"; df -h /home | tail -n 1
echo "== home pull"; tail -n 3 logs/pull_ckpts.log 2>&1; ls -la models/checkpoints/v4_cloud/ 2>&1 | tail -n 3; pgrep -af '^rsync' | cut -c1-140
true
