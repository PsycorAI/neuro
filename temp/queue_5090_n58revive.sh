#!/bin/bash
# queue_5090_n58revive.sh -- 2026-10-12 (Neuro 58; Glenn: "prep and schedule the test of your design to revive dead
# layers"). ONE job at a time on the 5090 (a 1B checkpoint is 10 GB in RAM; two at once started the 13:54 OOM loop):
#  1. stop the safety run (probe B, embedding decay off) -- its trainer only, SIGTERM
#  2. B @ 46,800: per-chunk perplexity (fixed seed), cross-chunk recall, layer diagnostic        = the control
#  3. temp/n58_revive.py: v4_cloud step 46,500 -> probe_v4_revive/step_0046500.pt (dead FFN units re-initialised,
#     function preserved)
#  4. arm C = the same config as B, from the revived checkpoint, to step 46,800 (then SIGTERM)
#  5. C @ 46,800: the same three evals
# Named queue_5090_* so temp/watchdog_tps.sh stays alive (started with WD_SKIP covering ^probe_).
# Each eval under a 40 min limit. Log: logs/n58_revive.log. One DONE / ALERT.
# Launch: setsid -f bash temp/queue_5090_n58revive.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/n58_revive.log
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HAYSTACK_B=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A(){ bash temp/alert.sh "$@"; }
say(){ echo "[revive $(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }
evals(){ # name ckpt
  say "evals of $1"
  timeout 2400 $PY temp/chunk_eval.py --T 2048 --iters 20 --B 4 $1=$2 2>&1 | grep -E 'C=|rror|Killed' | cut -c1-400 >> $LOG
  timeout 2400 $PY temp/xchunk_haystack_eval.py $1=$2 2>&1 | grep -E 'C=|rror|Killed' | cut -c1-300 >> $LOG
  timeout 2400 $PY temp/n58_layer_diag.py --B 2 $1=$2 2>&1 | grep -E 'blk|embed|rror|Killed' | cut -c1-200 >> $LOG
}
stop_trainer(){ # config name
  pkill -TERM -f "src/train_gpu.py --config configs/$1.yaml"
  for i in $(seq 1 90); do pgrep -f "src/train_gpu.py --config configs/$1.yaml" > /dev/null || return 0; sleep 2; done
  return 1
}
say "start"
B=models/checkpoints/probe_v4_nodecay/step_0046800.pt
stop_trainer probe_v4_nodecay || { A ALERT "n58 revive: the safety run's trainer did not stop"; exit 1; }
sleep 10
[ -f $B ] || { A ALERT "n58 revive: $B is missing"; exit 1; }
evals B46800 $B
say "revive"
timeout 2400 $PY temp/n58_revive.py models/checkpoints/v4_cloud_pin/step_0046500.pt models/checkpoints/probe_v4_revive/step_0046500.pt 2>&1 \
  | grep -E 'blk|logit|wrote|rror|Killed' | cut -c1-200 >> $LOG
[ -f models/checkpoints/probe_v4_revive/step_0046500.pt ] || { A ALERT "n58 revive: no revived checkpoint (logs/n58_revive.log)"; exit 1; }
$PY temp/n58_mk_probe.py probe_v4_revive eval_every=100 ckpt_every=100 'no_decay_names=[embed.weight]' >> $LOG 2>&1
say "arm C training"
setsid -f bash -c "exec $PY src/train_gpu.py --config configs/probe_v4_revive.yaml --device cuda >> logs/probe_v4_revive.log 2>&1" < /dev/null > /dev/null 2>&1
end=$(( $(date +%s) + 7200 ))
until [ -f models/checkpoints/probe_v4_revive/step_0046800.pt ]; do
  sleep 30
  pgrep -f "src/train_gpu.py --config configs/probe_v4_revive.yaml" > /dev/null || { A ALERT "n58 revive: arm C trainer died: $(grep -a -E '^[A-Za-z]*Error' logs/probe_v4_revive.log | tail -n 1 | cut -c1-140)"; exit 1; }
  [ "$(date +%s)" -ge "$end" ] && { A ALERT "n58 revive: arm C did not reach step 46,800 in 2 h"; stop_trainer probe_v4_revive; exit 1; }
done
sleep 20; stop_trainer probe_v4_revive; sleep 10
grep -a -E 'eval\]|\[mlp\]|\[wnorm\]' logs/probe_v4_revive.log | sed -E 's/ thr=[0-9.]+//g; s/rate=//g; s/\| target.*//' | paste - - - | cut -c1-230 >> $LOG
evals C46800 models/checkpoints/probe_v4_revive/step_0046800.pt
say "done"
A DONE "n58 revival test finished: logs/n58_revive.log | $(grep -E '^C46800 .*chunk1' $LOG | cut -c1-150)"
