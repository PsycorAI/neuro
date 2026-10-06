#!/bin/bash
# run_sftw4_5090.sh -- 2026-09-26 evening: finish rehearsal sweep after the G: outage.
# Trains w005 then w02s1 sequentially on the 5090 (+ single-pass recall after each),
# then tool_evals w02s1 here (the 5080 must be free by 19:00; it only evals w005).
export CUDA_VISIBLE_DEVICES=0
source /home/glenn/projects/neuro/temp/sftw_lib.sh
say "5090 runner v4 START (w005, w02s1)"
for a in w005 w02s1; do
  rm -rf models/checkpoints/screen_sftab_${a}_363m
  say "arm $a START"
  $PY src/train_sft.py --config configs/screen_sftab_${a}_363m.yaml --device cuda > logs/screen_sftab_${a}.log 2>&1
  say "arm $a exit=$?"
  $PY scripts/recall_eval.py --runs flagship_v3_363m,screen_sftab_${a}_363m --B 32 2>&1 | grep -E '^ *(run|flagship|screen|chance)' >> "$SLOG"
done
eval_arm w02s1
eval_arm w005   # fallback: no-op if the 5080 already claimed it
say "5090 DONE"
