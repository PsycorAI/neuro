#!/bin/bash
# queue_5080_toolresult.sh -- 2026-10-12 (Neuro 57, plan D3). Teacher data "tool result -> reply" on the 5080:
# held-out set first (small, 60 per tool), then the training set (1,000 per tool). Time limits on both. One DONE / ALERT.
# Launch: setsid -f bash temp/queue_5080_toolresult.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
PY=/home/glenn/projects/bdh/venv/bin/python; L=logs/gen_toolresult.log
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1
A(){ bash temp/alert.sh "$@"; }
A INFO "tool-result teacher data started on the 5080"
timeout 3600 $PY scripts/gen_toolresult_data.py --split test --per-tool 60 > $L 2>&1
timeout 14400 $PY scripts/gen_toolresult_data.py --split train --per-tool 1000 >> $L 2>&1
n=$(wc -l < training_data/posttrain/toolresult.jsonl 2>/dev/null || echo 0)
if [ "$n" -gt 3000 ]; then A DONE "tool-result data: $n train lines, $(wc -l < training_data/posttrain/toolresult_test.jsonl) test | $(grep '^done:' $L | tail -1) | $L"
else A ALERT "tool-result data: only $n lines: $(grep -E 'Error|Traceback' $L | tail -1 | cut -c1-140) ($L)"; fi
