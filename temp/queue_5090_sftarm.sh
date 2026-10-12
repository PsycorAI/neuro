#!/bin/bash
# queue_5090_sftarm.sh <run> [gpu=0] -- 2026-10-12 (Neuro 57, demo plan). One 13L SFT rehearsal arm: train
# configs/<run>.yaml (3 h limit, resumes from its own checkpoint), then tool_eval as trained and with all 9 tools
# offered (the demo setting), then the r25 evals. Every step has a time limit (a guard, not a schedule).
# Results: logs/<run>.log, logs/<run>_evals.log, one DONE / ALERT line. Name keeps the queue_5090_ prefix (watchdog_tps).
# Launch: setsid -f bash temp/queue_5090_sftarm.sh <run> 0 < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
R=${1:?run}; GPU=${2:-0}
PY=/home/glenn/projects/bdh/venv/bin/python; EV=logs/${R}_evals.log
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
A(){ bash temp/alert.sh "$@"; }
say(){ echo "[sftarm $(date '+%m-%d %H:%M:%S')] $*" >> $EV; }
touch logs/$R.log
A INFO "$R (13L SFT arm) started on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU timeout 10800 $PY src/train_sft.py --config configs/$R.yaml --device cuda >> logs/$R.log 2>&1
rc=$?
say "train exit=$rc | $(grep '^step' logs/$R.log | tail -1 | cut -c1-90)"
grep -q '^done:' logs/$R.log || { A ALERT "$R did NOT finish: $(grep -E 'Error|Traceback' logs/$R.log | tail -1 | cut -c1-140)"; exit 1; }
ck=$(ls models/checkpoints/$R/step_*.pt | tail -1)
CUDA_VISIBLE_DEVICES=$GPU timeout 2700 $PY scripts/tool_eval.py --run $R --n 200 --modes constrained > logs/${R}_tool_eval.log 2>&1
grep -E '^  constrained' logs/${R}_tool_eval.log | cut -c1-200 | sed "s/^/  tool_eval/" >> $EV
CUDA_VISIBLE_DEVICES=$GPU timeout 2700 $PY scripts/tool_eval.py --run $R --n 200 --modes constrained --all-tools > logs/${R}_tool_eval_all9.log 2>&1
grep -E '^  constrained' logs/${R}_tool_eval_all9.log | cut -c1-200 | sed "s/^/  tool_eval ALL-9/" >> $EV
GPU=$GPU T=2048 timeout 5400 bash temp/run_r25_evals.sh "$R=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $EV
say "evals DONE"
c1=$(grep "r25-evals .*chunk1" $EV | tail -1 | sed -n 's/.*| chunk1 *\([0-9.]*\).*/\1/p')
t1=$(grep 'tool_eval  ' $EV | tail -1 | sed -n 's/.*\(mode_acc.*\)(.*/\1/p' | cut -c1-110)
t9=$(grep 'tool_eval ALL-9' $EV | tail -1 | sed -n 's/.*\(tool_acc[^ ]* args_exact[^ ]*\).*/\1/p')
if [ -z "$t1" ]; then A ALERT "$R trained but tool_eval gave no result (see logs/${R}_tool_eval.log)"
else A DONE "$R | $t1| all-9: $t9 | chunk1 ${c1:-?} (reh13: 54.59; tool .959 args .803) | $EV"; fi
