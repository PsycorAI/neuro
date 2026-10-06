#!/usr/bin/env bash
# flagship_v3 post-training pipeline: SFT -> ORPO -> OPD, with the evals between.
#
# DO NOT RUN until flagship_v3_363m has reached its 10B ceiling. Each stage's
# *_init resolves a run_name to its LATEST checkpoint, so launching early
# silently post-trains a partial base. The guard below enforces it.
#
# Resumable: every stage auto-resumes from its own latest step_*.pt, and a
# .done marker skips completed stages. Safe to kill at any point.
#
#   GPU=0 bash temp/run_v3_posttrain.sh            # 5090 (after v3 finishes)
#   GPU=1 SKIP_OPD=1 bash temp/run_v3_posttrain.sh # 5080: SFT+ORPO only
#
# OPD needs the teacher resident alongside the student, which does not fit
# comfortably on the 5080's 16GB -- run that stage on the 5090.
set -u

cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
GPU=${GPU:-0}
SKIP_OPD=${SKIP_OPD:-0}
MIN_TOKENS=${MIN_TOKENS:-10900000000}   # healed base = 11B (2026-09-28)     # refuse to start on a partial base
LOG=logs/v3_posttrain.log
DONE=logs/v3_posttrain.done.d
mkdir -p "$DONE"
export CUDA_VISIBLE_DEVICES=$GPU
export TOKENIZERS_PARALLELISM=false

say() { echo "[pipe] $*" | tee -a "$LOG"; }

# ---- guard: is the base actually finished? --------------------------------
base_tok=$($PY - <<'PY'
import glob, torch
cks = sorted(glob.glob("/home/glenn/projects/neuro/models/checkpoints/"
                       "heal_v3_363m/step_*.pt"))   # 2026-09-28: healed base
print(0 if not cks else
      int(torch.load(cks[-1], map_location="cpu", weights_only=False).get("tokens", 0)))
PY
)
say "flagship_v3 latest ckpt = ${base_tok} tokens (need >= ${MIN_TOKENS})"
if [ "$base_tok" -lt "$MIN_TOKENS" ]; then
  say "REFUSING: base is not finished. Override with MIN_TOKENS=<n> if deliberate."
  exit 1
fi

stage() {   # stage <name> <script> <config>
  local name=$1 script=$2 cfg=$3
  if [ -f "$DONE/$name" ]; then say "$name already done -- skip"; return 0; fi
  say "=== $name ($cfg) on GPU $GPU $(date +%H:%M) ==="
  $PY "$script" --config "$cfg" --device cuda >> "$LOG" 2>&1
  local rc=$?
  say "$name exit=$rc"
  [ $rc -eq 0 ] && touch "$DONE/$name"
  return $rc
}

eval_stage() {   # eval_stage <run_name>
  local run=$1
  say "--- evals for $run ---"
  # B<=32: recall_eval at B=128 OOMs on the 5080 (neuro_recall_eval_b128_at_363m)
  # single-pass (2026-09-26): split replay has no KV carry -> invalid on attention stacks.
  # The base row anchors how much recall each stage costs.
  $PY scripts/recall_eval.py --runs heal_v3_363m,"$run" --B 32 >> "$LOG" 2>&1
  $PY scripts/tool_eval.py --run "$run" --n 200 >> "$LOG" 2>&1
  $PY scripts/abstain_eval.py --run "$run" --n 150 >> "$LOG" 2>&1
}

stage sft  src/train_sft.py configs/sft_v3_363m.yaml  || exit 1
eval_stage sft_v3_363m

stage orpo src/train_dpo.py configs/orpo_v3_363m.yaml || exit 1
eval_stage orpo_v3_363m

RC=orpo_v3_363m                     # release candidate the heads attach to
if [ "$SKIP_OPD" = "0" ]; then
  stage opd src/train_opd.py configs/opd_v3_363m.yaml || exit 1
  eval_stage opd_v3_363m
  RC=opd_v3_363m
else
  say "SKIP_OPD=1 -- stopping after ORPO (run OPD on the 5090)"
fi

# ---- auxiliary heads on the FROZEN release candidate ----------------------
# Optional and cheap: the backbone is frozen, so these cannot destabilise the
# model and a bad head is deleted rather than a model retrained. Set
# SKIP_HEADS=1 to stop after the language stages.
if [ "${SKIP_HEADS:-0}" = "0" ]; then
  say "heads attach to release candidate: $RC"
  # head_init in both configs defaults to opd_v3_363m; override when OPD was skipped
  if [ "$RC" != "opd_v3_363m" ]; then
    say "NOTE: edit head_init in configs/head_*_v3_363m.yaml to $RC before running heads"
  else
    stage head_energy src/train_heads.py configs/head_energy_v3_363m.yaml
    $PY scripts/head_eval.py --run head_energy_v3_363m --head energy \
        --prefs training_data/posttrain/why.jsonl --n 200 >> "$LOG" 2>&1
    stage head_conf   src/train_heads.py configs/head_conf_v3_363m.yaml
    $PY scripts/head_eval.py --run head_conf_v3_363m --head confidence \
        --n 200 >> "$LOG" 2>&1
  fi
fi

say "=== pipeline END $(date) ==="
