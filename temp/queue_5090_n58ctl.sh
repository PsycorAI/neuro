#!/bin/bash
# queue_5090_n58ctl.sh -- 2026-10-12 (Neuro 58). The missing CLEAN control for the revival test: the same config as
# arms B / D (embedding decay off, recall training on), from the untouched step-46,500 checkpoint, ONE uninterrupted
# segment to step 46,800, then the same fixed-seed evals. (B @ 46,800 = chunk1 50.01 came out of a segment that was
# killed and resumed twice during the 13:54 incident; D, narrow revival, = 42.13.)
# Log: logs/n58_revive.log. One DONE / ALERT. Launch: setsid -f bash temp/queue_5090_n58ctl.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/n58_revive.log; R=probe_v4_ctl
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HAYSTACK_B=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A(){ bash temp/alert.sh "$@"; }
say(){ echo "[ctl $(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }
gate(){ until [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 13 ]; do sleep 30; done; }
stop_trainer(){ pkill -TERM -f "src/train_gpu.py --config configs/$1.yaml"
  for i in $(seq 1 90); do pgrep -f "src/train_gpu.py --config configs/$1.yaml" > /dev/null || return 0; sleep 2; done; return 1; }
mkdir -p models/checkpoints/$R; rm -f models/checkpoints/$R/step_*.pt
ln models/checkpoints/v4_cloud_pin/step_0046500.pt models/checkpoints/$R/ || exit 1
$PY temp/n58_mk_probe.py $R eval_every=100 ckpt_every=100 'no_decay_names=[embed.weight]' >> $LOG 2>&1
say "control training (no revival)"
gate
setsid -f bash -c "exec $PY src/train_gpu.py --config configs/$R.yaml --device cuda >> logs/$R.log 2>&1" < /dev/null > /dev/null 2>&1
end=$(( $(date +%s) + 7200 ))
until [ -f models/checkpoints/$R/step_0046800.pt ]; do
  sleep 30
  pgrep -f "src/train_gpu.py --config configs/$R.yaml" > /dev/null || { A ALERT "n58 ctl: trainer died: $(grep -a -E '^[A-Za-z]*Error' logs/$R.log | tail -n 1 | cut -c1-140)"; exit 1; }
  [ "$(date +%s)" -ge "$end" ] && { A ALERT "n58 ctl: did not reach step 46,800 in 2 h"; stop_trainer $R; exit 1; }
done
sleep 20; stop_trainer $R; sleep 10
grep -a -E 'eval\]|\[mlp\]|\[wnorm\]' logs/$R.log | sed -E 's/ thr=[0-9.]+//g; s/rate=//g; s/\| target.*//' | paste - - - | cut -c1-230 >> $LOG
say "evals of CTL46800"
gate; timeout 2400 $PY temp/chunk_eval.py --T 2048 --iters 20 --B 4 CTL46800=models/checkpoints/$R/step_0046800.pt 2>&1 | grep -E 'C=|rror|Killed' | cut -c1-400 >> $LOG
gate; timeout 2400 $PY temp/xchunk_haystack_eval.py CTL46800=models/checkpoints/$R/step_0046800.pt 2>&1 | grep -E 'C=|rror|Killed' | cut -c1-300 >> $LOG
say "done"
A DONE "n58 clean control finished: $(grep -E '^CTL46800 .*chunk1' $LOG | cut -c28-110) | narrow revival D 42.13, interrupted B 50.01"
