#!/bin/bash
# queue_5080_treval.sh <run> [n=25] [gpu=1] -- 2026-10-12 (Neuro 57, plan D3). Second-pass faithfulness of one run
# (scripts/toolresult_eval.py), 45 min limit, one DONE / ALERT line. Log: logs/<run>_toolresult_eval.log
# Launch: setsid -f bash temp/queue_5080_treval.sh sft_v4_reh13 25 1 < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
R=${1:?run}; N=${2:-25}; export CUDA_VISIBLE_DEVICES=${3:-1} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
timeout 2700 /home/glenn/projects/bdh/venv/bin/python scripts/toolresult_eval.py --run $R --n $N > logs/${R}_toolresult_eval.log 2>&1
s=$(grep '^  toolresult' logs/${R}_toolresult_eval.log | sed 's/.*'$R' //' | sed 's/ faithful / /' | tr -s ' ' | tr '\n' ';' | cut -c1-240)
if [ -n "$s" ]; then bash temp/alert.sh DONE "second pass, $R: $s"
else bash temp/alert.sh ALERT "second-pass eval of $R gave nothing: $(grep -E 'Error|Traceback' logs/${R}_toolresult_eval.log | tail -1 | cut -c1-140)"; fi
