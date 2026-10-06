#!/bin/bash
# sftw_lib.sh -- 2026-09-26 SFT rehearsal sweep round 3: shared helpers (sourced).
cd /home/glenn/projects/neuro
export PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
SLOG=logs/sftab_screen.log
CLAIM=logs/tooleval_n200.claims.d
mkdir -p "$CLAIM"
# eval order: existing arms first, then this round's
EVAL_ARMS="mix mixmqar02 mixmqar w03 w01 hin02 w02s1"
say(){ echo "[sftw $(date +%H:%M:%S) gpu$CUDA_VISIBLE_DEVICES] $*" | tee -a "$SLOG"; }
trained(){ grep -q "arm $1 exit=0" "$SLOG"; }
eval_arm(){   # $1 = tag; atomic claim via mkdir so the two GPUs never eval the same arm
  mkdir "$CLAIM/$1" 2>/dev/null || return 1
  say "tool_eval n200 $1 START"
  $PY scripts/tool_eval.py --run screen_sftab_$1_363m --n 200 --modes free > logs/tooleval_n200_$1.log 2>&1
  say "tool_eval n200 $1 exit=$? :: $(grep -E '^ *free +parse' logs/tooleval_n200_$1.log)"
}
eval_pass(){  # one pass over the list: eval every trained, unclaimed arm; return 0 if all claimed
  local left=0
  for a in $EVAL_ARMS; do
    [ -d "$CLAIM/$a" ] && continue
    if trained "$a"; then eval_arm "$a"; else left=1; fi
  done
  return $left
}
eval_until_done(){ until eval_pass; do sleep 60; done; }
