#!/bin/bash
# queue_5090_sepbw.sh -- 2026-10-11 (Neuro 55; Glenn: build and test the bigger-batch option).
# GPU validation of mqar_separate_backward (train_gpu: the LM loss is back-propagated before the MQAR aux forward, so a
# step's peak memory is max(LM, aux) instead of LM + aux). Runs on the 5090 between r26w and r26x: r26x / r26y are held
# as "#HOLD" lines in temp/queue_5090_n54.list, so the n54 chain exits after r26w; this script then waits (max 45 min)
# for temp/sepbw.READY (= the CPU equivalence test passed), runs the arms below, restores the list and runs the chain.
#  A. 1B dense, cross-chunk step, B1 T2048, flag ON       (flag off, 10-11 probe: peak 24.9 GB)
#  B. 1B dense, plain step,       B1 T2048, flag ON       (flag off: 21.7 GB)
#  C. 1B dense, plain step,       B2 T1024, flag ON       (same token count as B: does memory follow tokens or samples?)
#  D. 1B CONVERSION smoke at the REAL shape (configs/v4_smoke1b_conv.yaml + flag, steps 150 -> 210 from the kept seed):
#     this died at its first forward on 10-10/11 (device not ready); it should now fit the 5090.
# Launch: setsid -f bash temp/queue_5090_sepbw.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro
PYB=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log; A(){ bash temp/alert.sh "$@"; }
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 HF_HOME=/home/glenn/.cache/huggingface
while pgrep -f "^bash temp/queue_5090_n54.sh" > /dev/null; do sleep 20; done
for i in $(seq 1 135); do [ -f temp/sepbw.READY ] && break; sleep 20; done
if [ -f temp/sepbw.READY ]; then
  M=logs/sepbw.log; echo "== mqar_separate_backward on the 5090 $(date '+%m-%d %H:%M') (1B, GA1; vram = allocated between steps / reserved)" >> $M
  m(){ tag=$1; cfg=$2; bs=$3; shift 3
    timeout 1800 $PYB temp/profile_train.py --config configs/$cfg.yaml --tag $tag --noprof --wait 3 --active 6 --batch $bs --grad_accum 1 \
        --set mqar_separate_backward=true "$@" > logs/sepbw_$tag.log 2>&1
    echo "$tag: $(grep -o 'peak allocated.*' logs/sepbw_$tag.log | tail -1) | $(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/sepbw_$tag.log | tail -1) | $(grep -E '^\[prof.*tok/s' logs/sepbw_$tag.log | sed 's/.*= \([0-9,]* tok\/s\).*/\1/') $(grep -hE 'Error' logs/sepbw_$tag.log | tail -1 | cut -c1-90)" >> $M
    rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml; }
  LATE="mqar_gap_start_frac=0 mqar_gap_ramp_frac=0 mqar_xchunk_start_frac=0"
  A INFO "separate-backward validation started on the 5090"
  m sep_xchunk_T2048   v4_smoke1b 1 $LATE mqar_xchunk_frac=1.0
  m sep_plain_T2048    v4_smoke1b 1 mqar_gap_start_frac=2.0
  m sep_plain_T1024_B2 v4_smoke1b 2 mqar_gap_start_frac=2.0 block_size=1024
  # D. the real conversion smoke with the flag
  r=v4_smoke1b_conv_sepbw; SEED=models/checkpoints/v4_smoke1b_conv_keep/step_0000150.pt
  if [ -f $SEED ] && [ -f configs/$r.yaml ]; then
    rm -rf models/checkpoints/$r; mkdir -p models/checkpoints/$r; ln $SEED models/checkpoints/$r/; : > logs/$r.log
    timeout 3000 $PYB src/train_gpu.py --config configs/$r.yaml --device cuda >> logs/$r.log 2>&1
    rc=$?
    last=$(ls models/checkpoints/$r/step_*.pt 2>/dev/null | tail -1 | sed -n 's/.*step_0*\([0-9]*\)\.pt/\1/p')
    echo "conv_T2048_full_recipe: exit=$rc last ckpt step $last | val $(grep 'eval\] val_ppl' logs/$r.log | awk '{print $3}' | tr '\n' ' ')| $(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/$r.log | tail -1) | $(grep '^step' logs/$r.log | tail -2 | sed -n 's/.*| \([0-9,]*\) tok\/s.*/\1/p' | tr '\n' ' ')tok/s $(grep -hE 'Error' logs/$r.log | tail -1 | cut -c1-90)" >> $M
    rm -rf models/checkpoints/$r
    [ "$last" = 210 ] && A DONE "separate backward: 1B CONVERSION at the real T2048 shape with the full recipe now PASSES on the 5090 | $(grep '^sep_xchunk_T2048' $M | tail -1 | cut -c1-70) (was 24.9 GB)" \
                     || A ALERT "separate backward: 1B conversion smoke did NOT pass | $(tail -n 1 $M | cut -c1-200)"
  else A ALERT "separate backward: conversion seed or config missing"; fi
else
  A WARN "separate-backward validation skipped: temp/sepbw.READY did not appear in 45 min"
fi
sed -i -E 's/^#HOLD (r26[xy]_)/\1/' temp/queue_5090_n54.list
exec bash temp/queue_5090_n54.sh
