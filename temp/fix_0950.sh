#!/bin/bash
# fix_0950.sh -- stop the waiting RC queue, add the throughput bench before r26d, relaunch.
cd /home/glenn/projects/neuro
for p in $(pgrep -f "^bash temp/queue_5090_rc.sh$"); do echo "TERM $p"; kill -TERM $p; done
sleep 1
sed -i 's/^for v in A B C D; do/for v in A B C; do/; s/, D = B + mqar_once_per_step//' temp/bench_rc_tps.sh
sed -i 's/^SRC=models\/checkpoints\/r26c_13L_H6_chunk1024_150m\/step_0012208.pt$/GPU=0 bash temp\/bench_rc_tps.sh\nSRC=models\/checkpoints\/r26c_13L_H6_chunk1024_150m\/step_0012208.pt/' temp/queue_5090_rc.sh
grep -n 'bench\|^SRC\|for v' temp/queue_5090_rc.sh temp/bench_rc_tps.sh
bash -n temp/queue_5090_rc.sh && bash -n temp/bench_rc_tps.sh && echo SYNTAX_OK
setsid -f bash temp/queue_5090_rc.sh > /dev/null 2>&1
sleep 1; pgrep -af "queue_50"
