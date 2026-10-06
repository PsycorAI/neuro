#!/bin/bash
# run_sftab_tooleval.sh -- 2026-09-26: does MQAR rehearsal cost instruction/tool quality?
# Fixed tool_eval (shuffled test slice; both arms trained on split=train only) on the 5080
# alongside the 30M depth screen (~2-3GB inference). n=100 to keep it ~1h total.
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/sftab_tooleval.log
for r in screen_sftab_mix_363m screen_sftab_mixmqar_363m; do
  echo "[tooleval $(date +%H:%M:%S)] $r" >> "$LOG"
  $PY scripts/tool_eval.py --run $r --n 100 2>&1 | grep -vE 'transformers|encoder' >> "$LOG"
done
echo "[tooleval $(date +%H:%M:%S)] DONE" >> "$LOG"
