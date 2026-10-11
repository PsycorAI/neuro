#!/bin/bash
# status.sh [dense_run] [conv_run] -- 2026-10-11 (Neuro 55). One-shot state of the v4 run ON the instance, as key=value
# lines. Called over ssh by cloud/cloud_watch.sh at home every few minutes; also handy by hand.
cd ${ROOT:-/workspace/neuro}
D=${1:-v4_cloud}; C=${2:-v4_cloud_conv}
cur=$D; [ -f models/checkpoints/$D/DONE ] && cur=$C
lg=logs/$cur.log
echo "runner=$(pgrep -f 'cloud/run_v4.sh' > /dev/null && echo 1 || echo 0)"
echo "trainer=$(pgrep -f 'src/train_gpu.py' > /dev/null && echo 1 || echo 0)"
echo "phase=$cur"
echo "dense_done=$([ -f models/checkpoints/$D/DONE ] && echo 1 || echo 0)"
echo "all_done=$([ -f models/checkpoints/$C/DONE ] && echo 1 || echo 0)"
echo "log_age_min=$(( ($(date +%s) - $(stat -c %Y $lg 2>/dev/null || date +%s)) / 60 ))"
echo "disk_free_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)"
echo "step_line=$(grep -a '^step' $lg 2>/dev/null | tail -1)"
# median tok/s of the last 5 step lines since the latest (re)start
echo "tps_med=$(awk '/^resumed |^arch=/{n=0} /^step [0-9]+ \|/{a[++n]=$0} END{for(i=(n>5?n-4:1);i<=n;i++) print a[i]}' $lg 2>/dev/null \
  | sed -n 's/.*| \([0-9,]*\) tok\/s.*/\1/p' | tr -d , | sort -n | awk '{a[NR]=$1} END{if (NR) print a[int((NR+1)/2)]}')"
echo "val=$(grep -a 'eval\] val_ppl' $lg 2>/dev/null | tail -3 | awk '{print $3}' | tr '\n' ' ')"
echo "total_steps=$(grep -a '^arch=' $lg 2>/dev/null | tail -1 | sed -n 's/.*total_steps[^0-9]*\([0-9]*\).*/\1/p')"
echo "trip=$(grep -a -c 'TRIPPED' $lg 2>/dev/null)"
echo "runner_last=$(tail -n 1 logs/run_v4.log 2>/dev/null)"
echo "gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)"
