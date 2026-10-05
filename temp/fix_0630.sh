#!/bin/bash
# fix_0630.sh -- 2026-10-05: p25g spilled (plain MQAR batch 32 on 16 GB). Stop the 5080 chain, regen the proxies as
# B/2 x GA2, clear p25g's partial ckpts (different batch -> no resume), relaunch g, b, h, e, d, c.
cd /home/glenn/projects/neuro
for p in $(pgrep -f "bash temp/queue_5080_r25.sh"); do kill $p; done
for p in $(pgrep -f "bash temp/run_5080_sched.sh"); do kill $p; done
for p in $(pgrep -f "configs/screen_p25"); do kill -TERM $p; done
sleep 3
/home/glenn/projects/bdh/venv/bin/python temp/mk_r25_configs.py > /dev/null
grep -H 'batch_size:\|grad_accum:\|mqar_aug_batch:' configs/screen_p25g*.yaml configs/screen_p25b*.yaml | tr '\n' ' '; echo
rm -rf models/checkpoints/screen_p25g_6L_chunk512_x_fix models/checkpoints/screen_p25b_6L_chunk512
sed -i 's/^r25evals screen_p25f_6L_chunk512_x_nl ""$/# (p25f evals done)/' temp/queue_5080_r25.sh
setsid -f bash temp/queue_5080_r25.sh > /dev/null 2>&1
echo "[s1 $(date '+%m-%d %H:%M:%S')] p25g SPILLED (plain MQAR batch 32 -> 15.3 GB, 8k tok/s; watchdog kill). 6L proxies now B/2 x GA2 (aux batches halved per micro); chain relaunched: g, b, h, e, d, c" >> logs/gap_queue.log
sleep 3; ps -eo pid,args | grep -E "bash temp/(queue_5080|run_5080)|screen_p25" | grep -v grep | cut -c1-110
