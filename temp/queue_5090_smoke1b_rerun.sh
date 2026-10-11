#!/bin/bash
# queue_5090_smoke1b_rerun.sh -- 2026-10-10 (Neuro 55; Glenn: rerun the 1B smoke BEFORE r26v).
# The first smoke (16:38) died at ~step 105 after its planned resume: 17.5 / 30.8 GB (ckpt left on the GPU) -> WDDM
# cliff at the first gap batch. train_gpu now frees the ckpt and re-keys the MQAR RNGs on resume. Same smoke as
# queue_5090_smoke1b.sh step 1; replaces queue_5090_after_smoke.sh (ends by running the n54 chain = r26v).
# Launch: setsid -f bash temp/queue_5090_smoke1b_rerun.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro
export T=2048; PYB=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log; A(){ bash temp/alert.sh "$@"; }
while pgrep -f "^bash temp/queue_5090_smoke1b.sh" > /dev/null; do sleep 60; done
rm -rf models/checkpoints/v4_smoke1b models/checkpoints/v4_smoke1b_conv
[ -s logs/v4_smoke1b.log ] && mv logs/v4_smoke1b.log logs/v4_smoke1b_fail1.log
: > logs/v4_smoke1b.log; : > logs/v4_smoke1b_conv.log
echo "[n55 $(date '+%m-%d %H:%M:%S')] 5090: 1B smoke RERUN v4_smoke1b -> v4_smoke1b_conv (cloud/run_v4.sh)" >> $LOG
A INFO "1B smoke RERUN started on the 5090"
setsid -f bash temp/kill_at_step.sh v4_smoke1b 50 < /dev/null > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=0 ROOT=/home/glenn/projects/neuro PY=$PYB HF_HOME=/home/glenn/.cache/huggingface \
  bash cloud/run_v4.sh v4_smoke1b v4_smoke1b_conv
par=$(grep -o 'params=[0-9.]*M' logs/v4_smoke1b.log | tail -1)
tps=$(grep '^step' logs/v4_smoke1b.log | tail -3 | sed -n 's/.*| \([0-9,]*\) tok\/s.*/\1/p' | tr '\n' ' ')
vr=$(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/v4_smoke1b.log | tail -1); vrc=$(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/v4_smoke1b_conv.log | tail -1)
vd=$(grep 'eval\] val_ppl' logs/v4_smoke1b.log | awk '{print $3}' | tr '\n' ' ')
vc=$(grep 'eval\] val_ppl' logs/v4_smoke1b_conv.log | awk '{print $3}' | tr '\n' ' ')
res=$(grep -c '^resumed' logs/v4_smoke1b.log)
if [ -f models/checkpoints/v4_smoke1b_conv/DONE ]; then
  A DONE "1B SMOKE RERUN PASSED ($par): dense val $vd($vr, $tps tok/s, $res resume) -> seed -> conversion val $vc($vrc)"
else
  A ALERT "1B SMOKE RERUN FAILED ($par, dense val ${vd:-none}, $vr): $(tail -n 1 logs/run_v4.log | cut -c1-100) | $(grep -hE 'Error|Traceback' logs/v4_smoke1b.log logs/v4_smoke1b_conv.log | tail -1 | cut -c1-120)"
fi
rm -rf models/checkpoints/v4_smoke1b models/checkpoints/v4_smoke1b_conv        # 2 x ~13 GB smoke checkpoints
exec bash temp/queue_5090_n54.sh
