#!/bin/bash
# swap_r26s_r26t.sh -- 2026-10-07 09:20 (Neuro 53). Stop r26s (r25o seed 1, 5 min in) -> run r26t (r26c seed 1).
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
for p in $(pgrep -f "^bash temp/queue_5090_r26s.sh$"); do echo "TERM queue $p"; kill -TERM $p; done
for p in $(pgrep -f "^bash temp/run_gap_queue.sh r26s_13L_H6_seed1_150m$"); do echo "TERM runner $p"; kill -TERM $p; done
for p in $(pgrep -f "train_gpu.py --config configs/r26s_13L_H6_seed1_150m.yaml"); do echo "TERM train $p"; kill -TERM $p; done
sleep 15
pgrep -af "r26s" || echo "r26s gone"
$PY temp/mk_p26_configs.py | grep r26t
sed -e 's/queue_5090_r26s.sh -- 2026-10-07 07:35/queue_5090_r26t.sh -- 2026-10-07 09:20/' -e 's/r26s = r25o seed 1/r26t = r26c seed 1/' \
    -e 's/^r=r26s_13L_H6_seed1_150m/r=r26t_13L_H6_chunk1024_seed1_150m/' -e 's/run_r25_evals.sh/run_r25_evals.sh/' \
    -e 's/GPU=0 bash temp\/run_r25_evals/GPU=0 T=2048 bash temp\/run_r25_evals/' -e 's/r26s queue DONE/r26t queue DONE/' \
    temp/queue_5090_r26s.sh > temp/queue_5090_r26t.sh
grep -nE '^r=|T=2048|DONE' temp/queue_5090_r26t.sh
echo "[n53 $(date '+%m-%d %H:%M:%S')] r26s (r25o seed 1) stopped after 5 min -> r26t (r26c seed 1: robustness of the recommended chunk-1024 layout)" >> logs/gap_queue.log
setsid -f bash temp/queue_5090_r26t.sh > /dev/null 2>&1
sleep 2; pgrep -af "queue_5090|train_gpu"
