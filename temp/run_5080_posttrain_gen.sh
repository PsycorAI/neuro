#!/usr/bin/env bash
# Post-training data generation chain on the 5080 (device 1). The 5090 (device 0)
# is the live flagship_v3 pretrain -- this script must never touch it.
#
# Every target is append-only JSONL + a sidecar progress file + a .done marker,
# so this is safe to kill at ANY time: re-running resumes where it stopped and
# skips completed targets.  Kill with:  pkill -f gen_teacher_data
#
# Order is the user's priority order, cheapest-first inside it so an early stop
# still leaves whole datasets:
#   tools (#4) -> coldstart (#1 OPD) -> why (#5) -> abstain (#2) -> coldstart retry
#
# coldstart passes --require-llama: its only job is to pre-align the student with
# the teacher that will later score tokens during OPD, so a different-tokenizer
# teacher there would defeat the stage. If the Llama-3.2-3B-Instruct download is
# still in flight, coldstart skips and is retried at the end of the chain. The
# other targets only need natural text and take whatever teacher is cached.
#
#   bash temp/run_5080_posttrain_gen.sh            # 300 min global budget
#   GLOBAL_MIN=120 bash temp/run_5080_posttrain_gen.sh
set -u

cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
export CUDA_VISIBLE_DEVICES=1          # 5080 ONLY
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1

GLOBAL_MIN=${GLOBAL_MIN:-300}
LOG=logs/posttrain_gen.log
END=$(( $(date +%s) + GLOBAL_MIN * 60 ))

# Single-instance lock. Two of these ran concurrently once and both loaded a
# multi-GB teacher onto the same 16GB card: VRAM thrashed and neither made
# progress for six minutes at 100% GPU. flock makes that impossible.
LOCK=/tmp/neuro_posttrain_gen.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[chain] another generation chain holds $LOCK -- refusing to start" | tee -a "$LOG"
  exit 1
fi

remaining() { echo $(( (END - $(date +%s)) / 60 )); }

once() {  # once <target> <n> <batch> <cap_min> [extra args...]
  local tgt=$1 n=$2 batch=$3 cap=$4; shift 4
  local rem; rem=$(remaining)
  if [ "$rem" -le 2 ]; then
    echo "[chain] global budget exhausted, stopping before $tgt" | tee -a "$LOG"
    return 1
  fi
  # Per-target cap so one slow target cannot starve the ones after it. `why`
  # carries a whole reply in its prompt and ran ~40x slower per record than
  # `tools`; left uncapped it would have consumed the entire window and left
  # coldstart (the OPD cold start, top priority) with nothing.
  if [ "$cap" -gt 0 ] && [ "$rem" -gt "$cap" ]; then rem=$cap; fi
  echo "[chain] === $tgt (n=$n batch=$batch budget=${rem}min) $(date +%H:%M) ===" | tee -a "$LOG"
  $PY scripts/gen_teacher_data.py --target "$tgt" --n "$n" --batch "$batch" \
      --max-minutes "$rem" "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "[chain] $tgt exit=$rc $(date +%H:%M)" | tee -a "$LOG"
  return $rc
}

run() {   # run <target> <n> <batch> <cap_min> [extra args...] -- halves the
          # batch and retries in a FRESH process if the target did not complete.
          # In-process backoff handles clean OOMs, but a cuBLAS internal error
          # can poison the CUDA context, and only a new process clears that.
          # cap_min 0 = no per-target cap (use the whole remaining budget).
  local tgt=$1 n=$2 batch=$3 cap=$4; shift 4
  local b=$batch
  local deadline=$(( $(date +%s) + (cap > 0 ? cap : 100000) * 60 ))
  for _ in 1 2 3; do
    if [ -f "training_data/posttrain/${tgt}.jsonl.done" ]; then
      echo "[chain] $tgt .done -- skip" | tee -a "$LOG"
      return 0
    fi
    once "$tgt" "$n" "$b" "$cap" "$@" && continue
    [ "$(remaining)" -le 2 ] && return 0
    [ "$(date +%s)" -ge "$deadline" ] && {
      echo "[chain] $tgt hit its ${cap}min cap -- moving on" | tee -a "$LOG"
      return 0; }
    b=$(( b / 2 )); [ "$b" -lt 4 ] && b=4
    echo "[chain] $tgt retry at batch $b" | tee -a "$LOG"
  done
  return 0
}

echo "[chain] START $(date) | global budget ${GLOBAL_MIN} min" | tee -a "$LOG"
# Batch sizes MEASURED on this card with Llama-3.2-3B-Instruct
# (temp/bench_teacher_gen.py, 200 new tokens, tool-shaped prompts):
#     batch  64 ->  8.62 GiB peak,  759 queries/min
#     batch 128 ->  9.71 GiB peak, 2825 queries/min
#     batch 192 -> 11.63 GiB peak, 3338 queries/min
# 192 for the short-prompt targets; lower where each sequence carries more
# tokens (coldstart generates 320, `why` puts a whole reply in the prompt).
#
# RESCHEDULED 2026-09-20 13:0x after measuring real rates. tools (24000) and
# abstain (16000) are DONE. Observed: tools ~750 records/min, abstain ~1290/min,
# but `why` only ~40/min (a whole reply sits in its prompt, and each item needs
# 400 new tokens). Left uncapped, `why` would have eaten the whole remaining
# window and left coldstart -- the OPD cold start, the top-priority dataset --
# completely empty.
#
# So: coldstart FIRST with the larger slice, capped so `why` still gets a real
# one; targets reduced to sizes that can actually finish. Both are blended with
# much larger HF corpora downstream, so these are signal, not the whole diet.
#   coldstart 16000 records is a solid cold start (recipe paper used 200k for a
#   far larger model; ours is one blended SFT stage).
#   why 6000 pairs = ~3000 rewritten replies, against 50% of UltraFeedback.
run tools     24000 192   0
run abstain   16000 192   0
# coldstart batch 32, NOT 64/128. Measured on this card: batch 128 failed
# outright and batch 64 crawled at ~28 records/min with nvidia-smi showing
# 15.9/16.3 GB -- i.e. spilling into WDDM shared memory, which this project has
# repeatedly measured as halving throughput. The unobvious VRAM hog is the
# PREFILL LOGITS tensor: HF computes logits for every prompt position on the
# first forward, and at a 128k teacher vocab that is batch x prompt_len x 128256
# x 2 bytes (~2.5GB at batch 64). Staying in physical VRAM beats a bigger batch.
# 2026-09-20 13:5x: Glenn is not in a hurry -- let coldstart run to its full
# 16000 rather than truncating it, then hand the card back for the OPD GPU smoke
# before 18:00. `why` is deferred to a later 5080 window (it resumes from its
# progress sidecar whenever this chain is relaunched).
run coldstart 16000  32 215 --require-llama
# why RESTORED 2026-09-21 (dedicated overnight 5080 window). It was deferred on
# 2026-09-20 because it is the slow target (~40 rec/min, whole reply in-prompt +
# 400 new tokens) and would have starved coldstart; coldstart is now done, so
# why gets the window. batch 16 (below coldstart's 32) -- why prompts are longer,
# and the prefill-logits tensor (batch x prompt_len x 128256 x 2B) is the VRAM
# hog to stay under on the 16GB card. Resumes from its sidecar (128 done).
run why       6000  16   0
echo "[chain] END $(date)" | tee -a "$LOG"
wc -l training_data/posttrain/*.jsonl 2>/dev/null | tee -a "$LOG"
