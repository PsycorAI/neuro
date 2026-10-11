#!/bin/bash
# pre_r26v_hook.sh -- 2026-10-11 (Neuro 55). Run once on the 5090 by queue_5090_convsmoke.sh, before r26v.
#  1. Home test of cloud/bench_rc3.sh at the 7L rehearsal shape (mechanics only: 5 arms + the budget line).
#  2. 1B MEMORY PROBE (B1 x GA1, temp/profile_train.py): true peak allocated per kind of step, to size the batch for
#     the H200 (143 GB). v4_cloud.yaml asks for B16 at T2048; the 5090 smoke reserved ~27 GB at B1.
#     per-sample cost = peak - "allocated between steps" (weights + optimizer + grads [+ teacher]).
cd /home/glenn/projects/neuro
PYB=/home/glenn/projects/bdh/venv/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 HF_HOME=/home/glenn/.cache/huggingface
echo "== pre_r26v_hook $(date '+%m-%d %H:%M:%S')"
rm -f logs/bench_rc3.log
ROOT=/home/glenn/projects/neuro PY=$PYB DENSE=v4_reh CONV=v4_reh_conv WAIT=5 ACTIVE=8 FORCE_BENCH_DATA=1 timeout 900 bash cloud/bench_rc3.sh
mv logs/bench_rc3.log logs/bench_rc3_hometest.log
bash temp/alert.sh INFO "bench_rc3 home test (7L, 5090): $(grep -c 'exit=0' logs/bench_rc3_hometest.log) of 5 arms ok | $(grep 'margin 7%' logs/bench_rc3_hometest.log | cut -c1-110)"
M=logs/mem1b.log; echo "== 1B memory probe $(date '+%m-%d %H:%M') (B1 x GA1; vram = allocated between steps / reserved)" >> $M
m(){ tag=$1; cfg=$2; shift 2
  timeout 600 $PYB temp/profile_train.py --config configs/$cfg.yaml --tag $tag --noprof --wait 3 --active 4 --batch 1 --grad_accum 1 \
      --set "$@" > logs/mem1b_$tag.log 2>&1
  echo "$tag: $(grep -o 'peak allocated.*' logs/mem1b_$tag.log | tail -1) | $(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/mem1b_$tag.log | tail -1) | $(grep -E '^\[prof.*tok/s' logs/mem1b_$tag.log | sed 's/.*= \([0-9,]* tok\/s\).*/\1/') $(grep -hE 'Error' logs/mem1b_$tag.log | tail -1 | cut -c1-90)" >> $M
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml; }
LATE="mqar_gap_start_frac=0 mqar_gap_ramp_frac=0 mqar_xchunk_start_frac=0"
m d_plain_T2048  v4_smoke1b      mqar_gap_start_frac=2.0
m d_plain_T1024  v4_smoke1b      mqar_gap_start_frac=2.0 block_size=1024
m d_gap_T2048    v4_smoke1b      $LATE mqar_xchunk_frac=0
m d_xchunk_T2048 v4_smoke1b      $LATE mqar_xchunk_frac=1.0
m c_plain_T1024  v4_smoke1b_conv mqar_gap_start_frac=2.0 block_size=1024
bash temp/alert.sh INFO "1B memory probe done: $(grep -c 'peak allocated' $M) of 5 arms | $(grep '^d_plain_T2048' $M | tail -1 | cut -c1-90)"
