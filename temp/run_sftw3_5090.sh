#!/bin/bash
# run_sftw3_5090.sh -- 2026-09-26: w005 spilled on the 5080 (15.9GB, ~3x slower) -> train it on
# the 5090 instead. Waits for run_sftw_5090.sh to log "arm w02s1 exit=", stops that runner while
# its post-arm recall_eval runs (the recall python survives and finishes; the runner never reaches
# its eval phase), trains w005, then drains the shared eval queue (incl. w005).
export CUDA_VISIBLE_DEVICES=0
source /home/glenn/projects/neuro/temp/sftw_lib.sh
EVAL_ARMS="$EVAL_ARMS w005"
until grep -q "arm w02s1 exit=" "$SLOG"; do sleep 5; done
pkill -f '[r]un_sftw_5090.sh'
say "5090 runner v3 START (w005)"
while pgrep -f 'recall_eval.py --runs flagship_v3_363m,screen_sftab_[w]02s1' > /dev/null; do sleep 10; done
rm -rf models/checkpoints/screen_sftab_w005_363m
say "arm w005 START"
$PY src/train_sft.py --config configs/screen_sftab_w005_363m.yaml --device cuda > logs/screen_sftab_w005.log 2>&1
say "arm w005 exit=$?"
$PY scripts/recall_eval.py --runs flagship_v3_363m,screen_sftab_w005_363m --B 32 2>&1 | grep -E '^ *(run|flagship|screen|chance)' >> "$SLOG"
eval_until_done
say "5090 DONE"
