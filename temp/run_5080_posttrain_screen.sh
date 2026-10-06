#!/usr/bin/env bash
# run_5080_posttrain_screen.sh -- METHOD screen of the SFT -> ORPO post-training
# stack on the 5080, from a FROZEN copy of the PARTIAL flagship_v3 base (~7B tok).
# Purpose: de-risk the pipeline ~4 days before the base hits 10B. Results are
# METHOD VALIDATION, not release quality (the cosine LR has not reached its end).
#
# Safe-by-construction: distinct screen run_names + a FROZEN base snapshot, so
# this never touches the live run's checkpoints and the future FINAL post-training
# run can never resume from these partial-base checkpoints. OPD is excluded
# (teacher-resident, 13.7GB -> 5090-only).
#
# Resumable: each stage auto-resumes from its own latest ckpt; .done markers skip
# finished stages. Relaunch this script to resume.
#
# LAUNCH (only when the 5080 is free):
#   setsid bash temp/run_5080_posttrain_screen.sh >/dev/null 2>&1 </dev/null &
# WATCH:
#   tail -f /home/glenn/projects/neuro/logs/posttrain_screen.log
# PAUSE:
#   pgrep -f '[r]un_5080_posttrain_screen.sh' | xargs -r kill
#   pgrep -f '[t]rain_sft.py' | xargs -r kill ; pgrep -f '[t]rain_dpo.py' | xargs -r kill
set -u
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
export CUDA_VISIBLE_DEVICES=1              # 5080 ONLY -- never the 5090 flagship
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
LOG=logs/posttrain_screen.log
DONE=logs/posttrain_screen.done.d
FROZEN=models/checkpoints/flagship_v3_363m_frozen_screen
mkdir -p "$DONE"

say(){ echo "[ptscreen $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# --- freeze the base once (idempotent): copy the latest COMPLETE base ckpt -----
# sft_init resolves a run_name to its LATEST ckpt, so one file in this dir pins
# the base. Copying a completed step_*.pt while the base trains is safe.
if ls "$FROZEN"/step_*.pt >/dev/null 2>&1; then
  say "frozen base already present: $(ls "$FROZEN"/step_*.pt | head -1)"
else
  mkdir -p "$FROZEN"
  latest=$(ls -t models/checkpoints/flagship_v3_363m/step_*.pt | head -1)
  say "freezing base snapshot: $latest -> $FROZEN/"
  cp "$latest" "$FROZEN"/
fi

# --- derive screen configs from the REAL configs (never drift) -----------------
$PY temp/mk_screen_configs.py >> "$LOG" 2>&1 || { say "config gen FAILED"; exit 1; }

whyn=$(wc -l < training_data/posttrain/why.jsonl 2>/dev/null || echo 0)
[ "$whyn" -lt 1000 ] && say "WARNING: why.jsonl has only $whyn records -- ORPO's why-narration axis is under-powered (abstain 16k + UltraFeedback still drive the screen). Regenerate why before the FINAL run."

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee -a "$LOG"

stage(){ # $1 name  $2 script  $3 config
  local name=$1 script=$2 cfg=$3 rc
  if [ -f "$DONE/$name" ]; then say "$name already done -- skip"; return 0; fi
  say "=== $name START ($cfg) ==="
  $PY "$script" --config "$cfg" --device cuda >> "$LOG" 2>&1
  rc=$?
  say "$name exit=$rc"
  [ $rc -eq 0 ] && touch "$DONE/$name"
  return $rc
}

eval_stage(){ # $1 run_name
  local run=$1
  say "--- evals for $run ---"
  $PY scripts/recall_eval.py --replay split --runs "$run" --B 32 >> "$LOG" 2>&1 || say "recall_eval failed"
  $PY scripts/tool_eval.py --run "$run" --n 200 >> "$LOG" 2>&1 || say "tool_eval failed"
  $PY scripts/abstain_eval.py --run "$run" --n 150 >> "$LOG" 2>&1 || say "abstain_eval failed"
}

say "=== posttrain SCREEN START/RESUME (5080; SFT+ORPO; base=frozen partial) ==="
stage sft  src/train_sft.py temp/sft_v3_363m_screen.yaml  || { say "PAUSED at SFT"; exit 0; }
eval_stage sft_v3_363m_screen
stage orpo src/train_dpo.py temp/orpo_v3_363m_screen.yaml || { say "PAUSED at ORPO"; exit 0; }
eval_stage orpo_v3_363m_screen
say "=== posttrain SCREEN DONE (OPD is 5090-only, run on the final base after 10B) ==="
