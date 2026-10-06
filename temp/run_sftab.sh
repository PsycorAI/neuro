#!/bin/bash
# run_sftab.sh -- 2026-09-26: why did v3 SFT destroy recall (base 100/100/100/72/38 -> 62/69/22/0/0)?
# SEQUENTIAL on the 5090 (two 426M SFT arms concurrently spilled past 32GB -> WDDM crawl).
#   mix      = the v3 mix (tools+coldstart+alpaca/2)  -- reproduces the loss at 6M tok?
#   mixmqar  = mix + MQAR recall rehearsal (mqar_aug, w0.5) -- the candidate fix
#   [1B probe: temp/run_1b_probe.sh -- v4, ~30 min]
#   alpaca   = alpaca-only (the v1 recipe that kept recall) -- diagnostic
# All from the v3 10B base, 6M tok. Recall = single-pass recall_eval (split is invalid
# on attention stacks). Each arm resumes from its ckpt if re-run.
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/sftab_screen.log
say(){ echo "[sftab $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
arm(){
  say "arm $1 START"
  $PY src/train_sft.py --config configs/screen_sftab_$1_363m.yaml --device cuda > logs/screen_sftab_$1.log 2>&1
  say "arm $1 exit=$?"
  $PY scripts/recall_eval.py --runs flagship_v3_363m,screen_sftab_$1_363m --B 32 2>&1 | grep -E '^ *(run|flagship|screen|chance)' >> "$LOG"
}
say "START (sequential)"
arm mix
arm mixmqar
say "1B probe START"; bash temp/run_1b_probe.sh; say "1B probe exit=$?"
arm alpaca
say "DONE"
