#!/bin/bash
# run_sftw_5090.sh -- 2026-09-26 SFT rehearsal sweep round 3, 5090: train w03, w01, hin02, w02s1
# (6M tok each, sequential -- two SFT arms spill the 5090), single-pass recall after each, then
# help drain the shared n=200 tool-eval queue.
export CUDA_VISIBLE_DEVICES=0
source /home/glenn/projects/neuro/temp/sftw_lib.sh
say "5090 START"
for a in w03 w01 hin02 w02s1; do
  say "arm $a START"
  $PY src/train_sft.py --config configs/screen_sftab_${a}_363m.yaml --device cuda > logs/screen_sftab_$a.log 2>&1
  say "arm $a exit=$?"
  $PY scripts/recall_eval.py --runs flagship_v3_363m,screen_sftab_${a}_363m --B 32 2>&1 | grep -E '^ *(run|flagship|screen|chance)' >> "$SLOG"
done
eval_until_done
say "5090 DONE"
