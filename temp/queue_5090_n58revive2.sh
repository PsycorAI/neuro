#!/bin/bash
# queue_5090_n58revive2.sh -- 2026-10-12 (Neuro 58). Second revival chain, one job at a time on the 5090:
#  1. A/B on the same fixed-seed eval: probe A @ 46,700 (embedding decay ON) vs probe B @ 46,700 (decay OFF)
#     -> is the chunk-1 perplexity rise (42.4 -> 50.0 at B @ 46,800) from the recall training or from the fix?
#  2. arm D = NARROW revival: only blocks 2-5 (the layers whose FFN is fully dead), only 10% of their dead units
#     (~1,200 per layer = what a healthy layer keeps alive), from step 46,500 to 46,800, same config as B
#  3. D @ 46,800: the same evals as B / C in logs/n58_revive.log
# Log: logs/n58_revive.log. One DONE / ALERT. Launch: setsid -f bash temp/queue_5090_n58revive2.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/n58_revive.log; R=probe_v4_revive2
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HAYSTACK_B=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A(){ bash temp/alert.sh "$@"; }
say(){ echo "[revive2 $(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }
gate(){ until [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 13 ]; do sleep 30; done; }
chunk(){ gate; timeout 2400 $PY temp/chunk_eval.py --T 2048 --iters 20 --B 4 $1=$2 2>&1 | grep -E 'C=|rror|Killed' | cut -c1-400 >> $LOG; }
evals(){ say "evals of $1"; chunk $1 $2
  gate; timeout 2400 $PY temp/xchunk_haystack_eval.py $1=$2 2>&1 | grep -E 'C=|rror|Killed' | cut -c1-300 >> $LOG
  gate; timeout 2400 $PY temp/n58_layer_diag.py --B 2 $1=$2 2>&1 | grep -E 'blk|rror|Killed' | cut -c1-200 >> $LOG; }
stop_trainer(){ pkill -TERM -f "src/train_gpu.py --config configs/$1.yaml"
  for i in $(seq 1 90); do pgrep -f "src/train_gpu.py --config configs/$1.yaml" > /dev/null || return 0; sleep 2; done; return 1; }
say "start: A/B at 46,700"
chunk A46700 models/checkpoints/probe_v4_xchunk/step_0046700.pt
chunk B46700 models/checkpoints/probe_v4_nodecay/step_0046700.pt
say "narrow revive (blocks 2-5, 10% of dead units)"
gate; timeout 2400 $PY temp/n58_revive.py --blocks 2,3,4,5 --frac 0.1 models/checkpoints/v4_cloud_pin/step_0046500.pt models/checkpoints/$R/step_0046500.pt 2>&1 \
  | grep -E 'blk|reviving|logit|wrote|rror|Killed' | cut -c1-200 >> $LOG
[ -f models/checkpoints/$R/step_0046500.pt ] || { A ALERT "n58 revive2: no revived checkpoint (logs/n58_revive.log)"; exit 1; }
$PY temp/n58_mk_probe.py $R eval_every=100 ckpt_every=100 'no_decay_names=[embed.weight]' >> $LOG 2>&1
say "arm D training"
gate
setsid -f bash -c "exec $PY src/train_gpu.py --config configs/$R.yaml --device cuda >> logs/$R.log 2>&1" < /dev/null > /dev/null 2>&1
end=$(( $(date +%s) + 7200 ))
until [ -f models/checkpoints/$R/step_0046800.pt ]; do
  sleep 30
  pgrep -f "src/train_gpu.py --config configs/$R.yaml" > /dev/null || { A ALERT "n58 revive2: arm D trainer died: $(grep -a -E '^[A-Za-z]*Error' logs/$R.log | tail -n 1 | cut -c1-140)"; exit 1; }
  [ "$(date +%s)" -ge "$end" ] && { A ALERT "n58 revive2: arm D did not reach step 46,800 in 2 h"; stop_trainer $R; exit 1; }
done
sleep 20; stop_trainer $R; sleep 10
grep -a -E 'eval\]|\[mlp\]|\[wnorm\]' logs/$R.log | sed -E 's/ thr=[0-9.]+//g; s/rate=//g; s/\| target.*//' | paste - - - | cut -c1-230 >> $LOG
evals D46800 models/checkpoints/$R/step_0046800.pt
say "done"
A DONE "n58 narrow revival finished: logs/n58_revive.log | $(grep -E '^D46800 .*chunk1' $LOG | cut -c28-150) | B 50.01, C 72.31"
