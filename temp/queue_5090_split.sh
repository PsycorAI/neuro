#!/bin/bash
# queue_5090_split.sh -- 2026-10-11 (Neuro 55). 5090 slot between r26x and r26y (r26y is held as a "#HOLD" line; this
# script restores it and re-runs the n54 chain when done). All arms: 1B (configs/v4_smoke1b*.yaml), LM batch 1, GA1.
# PART 1 -- GPU sanity of mqar_aux_splits (~10 min), separate backward on, 2 aux sequences per step:
#   split_xchunk2  cross-chunk aux batch 2 in 2 parts -> expect the peak of ONE aux sequence (~21.4 GB, logs/sepbw.log),
#                  not of two (~29 GB = over the 5090's cliff, which is why the unsplit arm is not run here)
#   split_plain8   plain MQAR aux batch 8 in 2 parts  -> expect about the plain peak (17.8 GB)
# PART 2 -- throughput (Glenn 10-11: look for remaining bottlenecks before the H200 bench), ~45 min:
#   prof_dense / prof_conv   torch.profiler tables of a dense plain step and a conversion step (logs/speed1b_<arm>.log)
#   t_dense_default vs t_dense_maxauto     compile_mode default vs max-autotune-no-cudagraphs (flagship v3 used it)
#   t_conv_default  vs t_conv_kdcompile    the Llama teacher uncompiled (as now) vs torch.compile(mode=default)
#   (conversion arms at T1024: the conversion + aux does not fit the 5090 at T2048 without the new flags)
# Launch: setsid -f bash temp/queue_5090_split.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro
PYB=/home/glenn/projects/bdh/venv/bin/python; A(){ bash temp/alert.sh "$@"; }
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 HF_HOME=/home/glenn/.cache/huggingface
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
# ---- PART 2: throughput ----
S=logs/speed1b.log; echo "== 1B throughput arms on the 5090 $(date '+%m-%d %H:%M') (B1 x GA1; the 5080 trains alongside -> compare arms, not absolutes)" >> $S
t(){ tag=$1; cfg=$2; prof=$3; tmo=$4; shift 4
  timeout $tmo $PYB temp/profile_train.py --config configs/$cfg.yaml --tag $tag $prof --wait 4 --active 10 --batch 1 --grad_accum 1 \
      --set mqar_gap_start_frac=2.0 "$@" > logs/speed1b_$tag.log 2>&1
  echo "$tag: $(grep -E '^\[prof.*tok/s' logs/speed1b_$tag.log | sed 's/^\[prof [a-z_0-9]*\] //') | $(grep -o 'peak allocated [0-9.]* GB' logs/speed1b_$tag.log | tail -1) $(grep -hE 'Error' logs/speed1b_$tag.log | tail -1 | cut -c1-90)" >> $S
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml logs/prof_$tag.trace.json; }
t prof_dense        v4_smoke1b      ""        1800
t t_dense_default   v4_smoke1b      --noprof  1800
t t_dense_maxauto   v4_smoke1b      --noprof  3000 compile_mode=max-autotune-no-cudagraphs
t prof_conv         v4_smoke1b_conv ""        1800 block_size=1024
t t_conv_default    v4_smoke1b_conv --noprof  1800 block_size=1024
t t_conv_kdcompile  v4_smoke1b_conv --noprof  2400 block_size=1024 kd_compile=true kd_compile_mode=default
A DONE "1B throughput arms done (logs/speed1b.log): $(grep -c 'tok/s' $S) arms | $(grep -E '^t_dense_(default|maxauto)' $S | sed 's/: median step [0-9]* ms = /=/; s/ (B1.*//' | tr '\n' ' ')"
sed -i -E 's/^#HOLD (r26y_)/\1/' temp/queue_5090_n54.list
exec bash temp/queue_5090_n54.sh
