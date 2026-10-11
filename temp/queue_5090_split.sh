#!/bin/bash
# queue_5090_split.sh -- 2026-10-11 (Neuro 55). GPU sanity of mqar_aux_splits on the 5090, between r26x and r26y
# (r26y is held as a "#HOLD" line; this script restores it and re-runs the n54 chain when done). ~10 min.
# 1B, LM batch 1 x T2048, mqar_separate_backward on, 2 aux sequences per step:
#   split_xchunk2  cross-chunk aux batch 2 in 2 parts -> expect the peak of ONE aux sequence (~21.4 GB, logs/sepbw.log),
#                  not of two (~29 GB = over the 5090's cliff, which is why the unsplit arm is not run here)
#   split_plain8   plain MQAR aux batch 8 in 2 parts  -> expect about the plain peak (17.8 GB)
# Launch: setsid -f bash temp/queue_5090_split.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro
PYB=/home/glenn/projects/bdh/venv/bin/python; A(){ bash temp/alert.sh "$@"; }
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "^bash temp/queue_5090_n54.sh" > /dev/null; do sleep 20; done
M=logs/sepbw.log; echo "== mqar_aux_splits 2 on the 5090 $(date '+%m-%d %H:%M') (1B, LM B1 x T2048, separate backward)" >> $M
m(){ tag=$1; shift
  timeout 1800 $PYB temp/profile_train.py --config configs/v4_smoke1b.yaml --tag $tag --noprof --wait 3 --active 6 --batch 1 --grad_accum 1 \
      --set mqar_separate_backward=true mqar_aux_splits=2 "$@" > logs/sepbw_$tag.log 2>&1
  echo "$tag: $(grep -o 'peak allocated.*' logs/sepbw_$tag.log | tail -1) | $(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/sepbw_$tag.log | tail -1) | $(grep -E '^\[prof.*tok/s' logs/sepbw_$tag.log | sed 's/.*= \([0-9,]* tok\/s\).*/\1/') $(grep -hE 'Error' logs/sepbw_$tag.log | tail -1 | cut -c1-90)" >> $M
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml; }
m split_xchunk2 mqar_gap_start_frac=0 mqar_gap_ramp_frac=0 mqar_xchunk_start_frac=0 mqar_xchunk_frac=1.0 mqar_xchunk_batch=2
m split_plain8  mqar_gap_start_frac=2.0 mqar_aug_batch=8
n=$(tail -n 2 $M | grep -c 'peak allocated')
[ "$n" = 2 ] && A DONE "aux splits on the 5090 (1B): $(tail -n 2 $M | cut -c1-60 | tr '\n' ' ')" || A ALERT "aux-splits sanity: an arm failed | $(tail -n 2 $M | cut -c1-150 | tr '\n' ' ')"
sed -i -E 's/^#HOLD (r26y_)/\1/' temp/queue_5090_n54.list
exec bash temp/queue_5090_n54.sh
