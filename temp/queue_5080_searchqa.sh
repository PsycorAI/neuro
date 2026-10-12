#!/bin/bash
# queue_5080_searchqa.sh -- 2026-10-12 (Neuro 57, plan D2 + D3 fix). 1. build the offline search pack (Simple English
# Wikipedia first paragraphs, SQLite FTS5); 2. search second-pass data on the 5080: the teacher writes questions FROM
# pack passages (test: 100 passages, train: 1,200 passages x 3 questions). Time limits; one DONE / ALERT.
# Launch: setsid -f bash temp/queue_5080_searchqa.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
PY=/home/glenn/projects/bdh/venv/bin/python; L=logs/gen_searchqa.log
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1
[ -f data/demo/search_pack.db ] || timeout 3600 nice $PY scripts/demo/build_search_pack.py > logs/build_search_pack.log 2>&1
[ -f data/demo/search_pack.db ] || { bash temp/alert.sh ALERT "search pack NOT built: $(tail -n 1 logs/build_search_pack.log | cut -c1-160)"; exit 1; }
export HF_HUB_OFFLINE=1
timeout 1800 $PY scripts/gen_searchqa_data.py --split test --pack 100 > $L 2>&1
timeout 7200 $PY scripts/gen_searchqa_data.py --split train --pack 1200 >> $L 2>&1
n=$(wc -l < training_data/posttrain/toolresult_search.jsonl 2>/dev/null || echo 0)
if [ "$n" -gt 500 ]; then bash temp/alert.sh DONE "search pack: $(tail -n 1 logs/build_search_pack.log | cut -c1-90) | search second-pass data: $n train lines | $(grep '^done:' $L | tail -1)"
else bash temp/alert.sh ALERT "search second-pass data: only $n lines: $(grep -E 'Error|Traceback' $L | tail -1 | cut -c1-140)"; fi
