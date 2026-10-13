#!/bin/bash
# queue_5090_n58decay_ab.sh -- 2026-10-12 (Neuro 58). Is the embedding fix behind the cloud's higher perplexity after
# step 66k (fixed-seed chunk1 of the pulled checkpoints: 64.5k 37.45 -> 72k 40.83), or is it the lower FFN layers
# waking? Home A/B in the cloud's OWN regime (plain training, no recall batches: the probe arms had those on), from
# the untouched step-46,500 checkpoint, 1,000 steps each, one uninterrupted segment each:
#   P = weight decay on the embedding (the launch recipe)        Q = no_decay_names: [embed.weight] (what runs now)
# Fixed-seed chunk_eval at steps 47,000 and 47,500 of both. ~2.7 h per arm. Starts after the SFT speed measurement.
# Log: logs/n58_decay_ab.log. One DONE / ALERT. Launch: setsid -f bash temp/queue_5090_n58decay_ab.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/n58_decay_ab.log
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A(){ bash temp/alert.sh "$@"; }
say(){ echo "[ab $(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }
gate(){ until [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 13 ]; do sleep 30; done; }
stop_trainer(){ pkill -TERM -f "src/train_gpu.py --config configs/$1.yaml"
  for i in $(seq 1 90); do pgrep -f "src/train_gpu.py --config configs/$1.yaml" > /dev/null || return 0; sleep 2; done; return 1; }
while pgrep -f '^bash temp/queue_5090_n58sftspeed.sh' > /dev/null; do sleep 60; done
PLAIN="mqar_gap_start_frac=0.7 mqar_gap_ramp_frac=0.1 mqar_xchunk_start_frac=0.5 mqar_xchunk_frac=0.25 eval_every=250 ckpt_every=500 keep_last=3"
arm(){ # run_name label extra-override
  R=$1; D=models/checkpoints/$R
  mkdir -p $D; rm -f $D/step_*.pt; ln models/checkpoints/v4_cloud_pin/step_0046500.pt $D/ || return 1
  $PY temp/n58_mk_probe.py $R $PLAIN $3 >> $LOG 2>&1
  say "arm $2 training ($R)"; gate
  setsid -f bash -c "exec $PY src/train_gpu.py --config configs/$R.yaml --device cuda >> logs/$R.log 2>&1" < /dev/null > /dev/null 2>&1
  end=$(( $(date +%s) + 14400 ))
  until [ -f $D/step_0047500.pt ]; do
    sleep 60
    pgrep -f "src/train_gpu.py --config configs/$R.yaml" > /dev/null || { A ALERT "n58 decay A/B: arm $2 trainer died: $(grep -a -E '^[A-Za-z]*Error' logs/$R.log | tail -n 1 | cut -c1-140)"; return 1; }
    [ "$(date +%s)" -ge "$end" ] && { A ALERT "n58 decay A/B: arm $2 did not reach 47,500 in 4 h"; stop_trainer $R; return 1; }
  done
  sleep 20; stop_trainer $R; sleep 10
  grep -a -E 'eval\]|\[wnorm\]' logs/$R.log | paste - - | cut -c1-200 >> $LOG
  for s in 0047000 0047500; do
    gate; timeout 2400 $PY temp/chunk_eval.py --T 2048 --iters 20 --B 4 ${2}_$s=$D/step_$s.pt 2>&1 | grep -E 'C=|rror|Killed' | cut -c1-300 >> $LOG
  done
  rm -f $D/step_0046500.pt $D/step_0047000.pt                        # keep only the last one (10 GB)
}
say "start"
arm probe_v4_plainP P "" || exit 1
arm probe_v4_plainQ Q "no_decay_names=[embed.weight]" || exit 1
say "done"
A DONE "n58 decay A/B (plain training, 1,000 steps): P decay on $(grep -E '^P_0047500' $LOG | cut -c44-58) | Q decay off $(grep -E '^Q_0047500' $LOG | cut -c44-58) | logs/n58_decay_ab.log"
