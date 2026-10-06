#!/bin/bash
# run_sftab2.sh -- 2026-09-26 follow-on to run_sftab.sh (waits for it). MQAR rehearsal
# at w0.5 kept recall (N32/64 59/25 vs 34/3) but cost tool skill (tool_acc .877->.789,
# args .684->.544, n=100). Map the tradeoff for Glenn's pick of the full-SFT weight:
#   1) mixmqar02 = mix + MQAR w0.2 (6M tok) + single-pass recall
#   2) B8xGA4 SFT speed check (100 steps; same 32 seqs/step) for the real run
#   3) tool_eval n=100 on mixmqar02 and alpaca (same fixed slice as the other arms)
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/sftab_screen.log
say(){ echo "[sftab $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
while pgrep -f '[r]un_sftab.sh' > /dev/null; do sleep 30; done
say "arm mixmqar02 START"
$PY src/train_sft.py --config configs/screen_sftab_mixmqar02_363m.yaml --device cuda > logs/screen_sftab_mixmqar02.log 2>&1
say "arm mixmqar02 exit=$?"
$PY scripts/recall_eval.py --runs flagship_v3_363m,screen_sftab_mixmqar02_363m --B 32 2>&1 | grep -E '^ *(run|flagship|screen|chance)' >> "$LOG"
say "B8 speed check START"
rm -rf models/checkpoints/screen_sftspeed_b8_363m
$PY src/train_sft.py --config temp/screen_sftspeed_b8_363m.yaml --device cuda > logs/screen_sftspeed_b8.log 2>&1
say "B8 speed check exit=$? :: $(grep -E '^step' logs/screen_sftspeed_b8.log | tail -n 1)"
rm -rf models/checkpoints/screen_sftspeed_b8_363m
for r in screen_sftab_mixmqar02_363m screen_sftab_alpaca_363m; do
  echo "[tooleval $(date +%H:%M:%S)] $r" >> logs/sftab_tooleval.log
  $PY scripts/tool_eval.py --run $r --n 100 2>&1 | grep -vE 'transformers|encoder' >> logs/sftab_tooleval.log
done
echo "[tooleval $(date +%H:%M:%S)] DONE (round 2)" >> logs/sftab_tooleval.log
say "sftab2 DONE"
