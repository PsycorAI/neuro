#!/bin/bash
# queue_5090_n58sftspeed.sh -- 2026-10-12 (Neuro 58; demo plan step 1: "measure 1B SFT speed + memory on the 5090").
# Waits until the checkpoint health check is done and >= 14 GB RAM is available (one 10 GB load at a time), then runs
# configs/sft_v4_dense_speed.yaml (1.5M tok, ~90 steps) under a 60 min limit and reports tok/s + VRAM. The run's own
# checkpoint dir is removed at the end (10 GB, not a model to keep). Log: logs/sft_v4_dense_speed.log. One DONE / ALERT.
# Launch: setsid -f bash temp/queue_5090_n58sftspeed.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
PY=/home/glenn/projects/bdh/venv/bin/python; R=sft_v4_dense_speed
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A(){ bash temp/alert.sh "$@"; }
$PY temp/n58_mk_sft_speed.py > logs/${R}_mk.log 2>&1 || { A ALERT "$R: config generation failed (logs/${R}_mk.log)"; exit 1; }
end=$(( $(date +%s) + 7200 ))
until ! pgrep -f '^bash temp/n58_ckpt_health.sh' > /dev/null && [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 14 ]; do
  [ "$(date +%s)" -ge "$end" ] && { A ALERT "$R: no free RAM in 2 h"; exit 1; }; sleep 30; done
rm -rf models/checkpoints/$R; : > logs/$R.log
timeout -s TERM 3600 $PY src/train_sft.py --config configs/$R.yaml --device cuda >> logs/$R.log 2>&1
rc=$?
tps=$(grep -a -E '^step' logs/$R.log | tail -n 8 | sed -n 's/.*| \([0-9,]*\) tok\/s.*/\1/p' | tr -d ',' | sort -n | awk '{a[NR]=$1} END{if (NR) print a[int((NR+1)/2)]}')
vr=$(grep -a -o 'vram [0-9.]*/[0-9.]* GB' logs/$R.log | tail -n 1)
if [ -n "$tps" ]; then A DONE "1B dense SFT on the 5090 (B4 x GA8 x 512): median $tps tok/s, $vr (rc $rc) -> 60M tok = $(( 60000000 / tps / 360 ))e-1 h | logs/$R.log"
else A ALERT "$R gave no step lines (rc $rc): $(grep -a -E 'Error|Traceback' logs/$R.log | tail -n 1 | cut -c1-160)"; fi
rm -rf models/checkpoints/$R
