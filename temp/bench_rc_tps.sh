#!/bin/bash
# bench_rc_tps.sh -- 2026-10-07 (Neuro 53, throughput weakness of the RC). 420-step training benches of the r26c layout on
# one free GPU: A = as is (B4xGA2), B = B8xGA1 (same tokens/step), C = A + mqar_once_per_step.
# xchunk is on from step ~126 and gap MQAR from ~294, so steps 300-420 carry every aux. Median tok/s of those steps.
# Usage: GPU=0 bash temp/bench_rc_tps.sh   -> logs/gap_queue.log (+ logs/bench_rc_*.log)
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
for v in A B C; do
  $PY - "$v" << 'PYEOF'
import sys, yaml
v = sys.argv[1]
c = yaml.safe_load(open("configs/r26c_13L_H6_chunk1024_150m.yaml"))
c.update(run_name=f"bench_rc_{v}", max_tokens=420 * 16384, eval_every=10**9, ckpt_every=10**9, log_every=20)
if v in "BD":
    c.update(batch_size=8, grad_accum=1)
if v in "CD":
    c.update(mqar_once_per_step=True)
yaml.safe_dump(c, open(f"temp/bench_rc_{v}.yaml", "w"), sort_keys=False)
PYEOF
  rm -rf models/checkpoints/bench_rc_$v
  $PY src/train_gpu.py --config temp/bench_rc_$v.yaml --device cuda > logs/bench_rc_$v.log 2>&1
  med=$(grep '^step' logs/bench_rc_$v.log | awk '{s=$2} s>=300' | sed -n 's/.*| \([0-9,]*\) tok\/s.*/\1/p' | tr -d ',' \
        | sort -n | awk '{a[NR]=$1} END{if (NR) print a[int((NR+1)/2)]}')
  vram=$(grep '^step' logs/bench_rc_$v.log | tail -1 | sed -n 's/.*vram \(.*\)/\1/p')
  echo "[bench $(date '+%m-%d %H:%M:%S')] RC tok/s $v ($(grep -E '^(batch_size|grad_accum|mqar_once_per_step):' temp/bench_rc_$v.yaml | tr '\n' ' ')): median ${med:-FAILED} (steps 300-420) | vram $vram" >> logs/gap_queue.log
  rm -rf models/checkpoints/bench_rc_$v temp/bench_rc_$v.yaml
done
