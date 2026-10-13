#!/bin/bash
# queue_5080_brainon.sh [gpu=1] -- 2026-10-12 (Neuro 58; demo plan D7 / section 5 "brain" row: tool score within 2
# points of no-brain, energy within 5%). 13L: the base SFT model vs the same model with each brain merged in
# (scripts/merge_brain.py): tool calling with all 9 tools offered (scripts/tool_eval.py, constrained, n 200) and
# energy-v2. Each step under a 45 min limit. Log: logs/n58_brainon.log. One DONE / ALERT.
# Launch: setsid -f bash temp/queue_5080_brainon.sh 1 < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
GPU=${1:-1}; BASE=sft_v4_reh13_v2tr; LOG=logs/n58_brainon.log
PY=/home/glenn/projects/bdh/venv/bin/python
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
: > $LOG
one(){ # label run
  echo "== $1 ($2) $(date +%H:%M)" >> $LOG
  timeout 2700 $PY scripts/tool_eval.py --run $2 --n 200 --modes constrained --all-tools 2>&1 | grep -E '^  constrained|^tool_acc|^args_exact|^mode_acc|^refuse_acc|^halluc|rror' | cut -c1-200 >> $LOG
  ck=$(ls models/checkpoints/$2/step_*.pt | tail -n 1)
  timeout 2700 $PY temp/energy_eval_v2.py $1=$ck 2>&1 | grep -E 'energy-v2|rror' | cut -c1-260 >> $LOG
}
one base $BASE
for b in py:brain_py13 health:brain_fdc13; do
  l=${b%%:*}; r=${b##*:}
  timeout 1200 $PY scripts/merge_brain.py --base $BASE --brain $r --out merged_$r 2>&1 | grep -E 'OK|rror' | cut -c1-200 >> $LOG
  one $l merged_$r
  rm -rf models/checkpoints/merged_$r
done
echo "== done $(date +%H:%M)" >> $LOG
bash temp/alert.sh DONE "brain ON vs OFF at 13L (tool score all-9, energy): $(grep -E '^== |^  constrained|energy-v2' $LOG | sed -E 's/ +/ /g' | cut -c1-110 | tr '\n' ';' | cut -c1-400) | logs/n58_brainon.log"
