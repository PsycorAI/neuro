#!/bin/bash
# queue_5090_n58resume.sh -- 2026-10-12 (Neuro 58). RESUME TEST at 1B. The clean control reached chunk1 42.47 at step
# 46,800 in one segment; the safety run B, whose last 100 steps followed a kill + resume from its own step-46,700
# checkpoint, reached 50.01. Is a resume from a checkpoint saved with no_decay_names (or any resume at 1B) harmful?
# Here: move the control's step_0046800 aside, RESUME the control from its step_0046700 checkpoint, run to 46,800
# again, same fixed-seed eval. Expected if the resume is clean: ~42.5.
# Log: logs/n58_revive.log. One DONE / ALERT. Launch: setsid -f bash temp/queue_5090_n58resume.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/n58_revive.log; R=probe_v4_ctl; D=models/checkpoints/$R
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HAYSTACK_B=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A(){ bash temp/alert.sh "$@"; }
say(){ echo "[resume $(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }
gate(){ until [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 13 ]; do sleep 30; done; }
stop_trainer(){ pkill -TERM -f "src/train_gpu.py --config configs/$1.yaml"
  for i in $(seq 1 90); do pgrep -f "src/train_gpu.py --config configs/$1.yaml" > /dev/null || return 0; sleep 2; done; return 1; }
[ -f $D/step_0046700.pt ] && [ -f $D/step_0046800.pt ] || { A ALERT "n58 resume test: control checkpoints missing"; exit 1; }
mkdir -p $D/keep; mv $D/step_0046800.pt $D/keep/first_segment_step_0046800.pt
say "resume the control from step 46,700"
gate
setsid -f bash -c "exec $PY src/train_gpu.py --config configs/$R.yaml --device cuda >> logs/$R.log 2>&1" < /dev/null > /dev/null 2>&1
end=$(( $(date +%s) + 5400 ))
until [ -f $D/step_0046800.pt ]; do
  sleep 30
  pgrep -f "src/train_gpu.py --config configs/$R.yaml" > /dev/null || { A ALERT "n58 resume test: trainer died: $(grep -a -E '^[A-Za-z]*Error' logs/$R.log | tail -n 1 | cut -c1-140)"; exit 1; }
  [ "$(date +%s)" -ge "$end" ] && { A ALERT "n58 resume test: did not reach step 46,800 in 90 min"; stop_trainer $R; exit 1; }
done
sleep 20; stop_trainer $R; sleep 10
grep -a -E '^resumed|remapped|warn' logs/$R.log | tail -n 2 | cut -c1-160 >> $LOG
grep -a -E 'eval\]|\[wnorm\]' logs/$R.log | tail -n 2 | paste - - | cut -c1-200 >> $LOG
gate; timeout 2400 $PY temp/chunk_eval.py --T 2048 --iters 20 --B 4 CTLR46800=$D/step_0046800.pt 2>&1 | grep -E 'C=|rror|Killed' | cut -c1-400 >> $LOG
say "done"
A DONE "n58 resume test: after a resume $(grep -E '^CTLR46800 .*chunk1' $LOG | cut -c44-68) | one segment 42.47 | interrupted B 50.01"
