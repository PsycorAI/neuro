#!/bin/bash
# fix_2315.sh -- 2026-10-08 23:15 (Neuro 53). Watchdog killed r26m (3.8k tok/s, CPU contention from the demo server);
# the 5080 follow-up waited on r26m's evals -> idle. Watchdog MIN_TPS 2500 (spills run 0.7-1.6k); replace the follow-up
# queue: 5080 resume r26m -> resume r26k; 5090 r26n after r26l (unchanged logic).
cd /home/glenn/projects/neuro
for p in $(pgrep -f "^bash temp/queue_after_night_1008.sh$"); do echo "TERM $p"; kill -TERM $p; done
for p in $(pgrep -f "^bash temp/watchdog_tps.sh$"); do echo "TERM watchdog $p"; kill -TERM $p; done
sed -e 's/^# queue_after_night_1008.sh.*/# queue_after_night2_1008.sh -- 2026-10-08 23:15 (Neuro 53): 5080 resume r26m -> resume r26k; 5090 r26n after r26l./' \
    -e "s/^  until grep -q 'r26m_13L_RC2_hola256_150m.*/  for rr in r26m_13L_RC2_hola256_150m r26k_13L_RC2_hola_seed1_150m; do/" \
    -e 's/^  r=r26k_13L_RC2_hola_seed1_150m$/  r=$rr/' \
    -e 's/^  \[ -f models\/checkpoints\/\$r\/DONE \] \&\& post 1 \$r$/  [ -f models\/checkpoints\/$r\/DONE ] \&\& post 1 $r; done/' \
    temp/queue_after_night_1008.sh > temp/queue_after_night2_1008.sh
bash -n temp/queue_after_night2_1008.sh || exit 1
sed -n '12,20p' temp/queue_after_night2_1008.sh
MIN_TPS=2500 setsid -f bash temp/watchdog_tps.sh > /dev/null 2>&1
setsid -f bash temp/queue_after_night2_1008.sh > /dev/null 2>&1
echo "[n53 $(date '+%m-%d %H:%M:%S')] r26m killed by watchdog (3.8k tok/s; demo-server CPU contention) -> 5080 resumes r26m (step 9000) then r26k (step 9000); watchdog MIN_TPS 2500. r26g2 (RC2 + Llama-1B KD conversion) = chunk1 37.48 (r26j 38.49, r21b 38.84) = NEW BEST, 5.56x / 3.16x SRAM, xchunk ~100." >> logs/gap_queue.log
sleep 90; ps -eo args | grep -E "queue_after|sched|src/train_gpu.py|watchdog" | grep -v grep | cut -c1-110
