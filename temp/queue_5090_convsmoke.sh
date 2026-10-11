#!/bin/bash
# queue_5090_convsmoke.sh -- 2026-10-11 00:00 (Neuro 55). 1B CONVERSION smoke at shapes that fit the 5090, run before
# r26v (held as "#HOLD" lines in temp/queue_5090_n54.list by temp/hold_convsmoke.sh). Configs: temp/mk_convsmoke.py.
#  A. v4_smoke1b_conv_t1024: full conversion recipe at T1024, one planned stop + resume after step 400 (4 attempts).
#  B. v4_smoke1b_conv_noaux: real T2048 shape without the MQAR aux, one attempt (does the main path fit 32 GB?).
# Then: restore the list and run the n54 chain (r26v, r26w, r26x). The *_keep checkpoint dirs stay until Glenn is told.
# Launch: setsid -f bash temp/queue_5090_convsmoke.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro
PYB=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log; A(){ bash temp/alert.sh "$@"; }
SEED=models/checkpoints/v4_smoke1b_conv_keep/step_0000150.pt
while pgrep -f "^bash temp/queue_5090_(smoke1b_rerun|n54).sh" > /dev/null || pgrep -f "^bash cloud/run_v4.sh" > /dev/null \
      || pgrep -f "src/train_gpu.py --config configs/v4_smoke1b" > /dev/null; do sleep 20; done
last(){ ls models/checkpoints/$1/step_*.pt 2>/dev/null | tail -1 | sed -n 's/.*step_0*\([0-9]*\)\.pt/\1/p'; }
run(){ r=$1; tries=$2; target=$3            # success = the last checkpoint is at the target step
  rm -rf models/checkpoints/$r; mkdir -p models/checkpoints/$r; ln $SEED models/checkpoints/$r/; : > logs/$r.log
  for t in $(seq 1 $tries); do
    echo "[n55 $(date '+%m-%d %H:%M:%S')] [$r] conversion smoke, attempt $t" >> $LOG
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
      HF_HOME=/home/glenn/.cache/huggingface timeout 3000 $PYB src/train_gpu.py --config configs/$r.yaml --device cuda >> logs/$r.log 2>&1
    rc=$?        # read it before any $(...): "exit=$?" after a $(date) prints date's status (queue_spk_conv.sh: always 0)
    echo "[n55 $(date '+%m-%d %H:%M:%S')] [$r] attempt $t exit=$rc last ckpt step $(last $r)" >> $LOG
    [ "$(last $r)" = "$target" ] && return 0
    sleep 45
  done; return 1; }
sum(){ r=$1; echo "val $(grep 'eval\] val_ppl' logs/$r.log | awk '{print $3}' | tr '\n' ' ')| $(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/$r.log | tail -1) | $(grep '^step' logs/$r.log | tail -3 | sed -n 's/.*| \([0-9,]*\) tok\/s.*/\1/p' | tr '\n' ' ')tok/s | $(grep -c '^resumed' logs/$r.log) resumes | $(grep -hE 'Error' logs/$r.log | tail -1 | cut -c1-90)"; }
if [ -f $SEED ] && [ -f configs/v4_smoke1b_conv_t1024.yaml ]; then
  A INFO "1B conversion smoke (T1024) started on the 5090"
  setsid -f bash temp/kill_at_step.sh v4_smoke1b_conv_t1024 400 < /dev/null > /dev/null 2>&1
  if run v4_smoke1b_conv_t1024 4 750; then A DONE "1B CONVERSION SMOKE at T1024 PASSED (steps 150 -> 750, gap phase from 630): $(sum v4_smoke1b_conv_t1024)"
  else A ALERT "1B CONVERSION SMOKE at T1024 FAILED: $(sum v4_smoke1b_conv_t1024)"; fi
  pkill -f "^bash temp/kill_at_step.sh v4_smoke1b_conv_t1024" 2>/dev/null
  if run v4_smoke1b_conv_noaux 1 210; then A INFO "1B conversion at the real T2048 shape WITHOUT the MQAR aux fits the 5090: $(sum v4_smoke1b_conv_noaux)"
  else A INFO "1B conversion at T2048 does not fit the 5090 even without the MQAR aux: $(sum v4_smoke1b_conv_noaux)"; fi
  rm -rf models/checkpoints/v4_smoke1b_conv_t1024 models/checkpoints/v4_smoke1b_conv_noaux
else A ALERT "conversion smoke: seed checkpoint or config missing, skipped"; fi
# Optional extra 5090 work before r26v (written while this driver runs, e.g. a home test of cloud/bench_rc3.sh).
[ -f temp/pre_r26v_hook.sh ] && timeout 2400 bash temp/pre_r26v_hook.sh >> logs/pre_r26v_hook.log 2>&1
sed -i -E 's/^#HOLD (r26[vwx]_)/\1/' temp/queue_5090_n54.list
exec bash temp/queue_5090_n54.sh
