#!/bin/bash
# fix_0535.sh -- 2026-10-05: kill the spilled p25f, restart the watchdog with the faster trigger, regen configs, queue r25c.
cd /home/glenn/projects/neuro
/home/glenn/projects/bdh/venv/bin/python temp/mk_r25_configs.py | grep -E 'p25h|r25c'
for p in $(pgrep -f "configs/screen_p25f_6L_chunk512_x_nl.yaml"); do kill -TERM $p; done
for p in $(pgrep -f "bash temp/watchdog_tps.sh"); do kill $p; done
sleep 2
setsid -f bash temp/watchdog_tps.sh > /dev/null 2>&1
setsid -f bash temp/queue_5090_r25c.sh > /dev/null 2>&1
echo "[s1 $(date '+%m-%d %H:%M:%S')] p25f KILLED: WDDM spill (17.1 GB on 16 GB, 2.4k tok/s); watchdog now 5 lines past step 300; p25h = B8xGA2; NL-fact test moved to 5090 r25c (after r25b)" >> logs/gap_queue.log
sleep 3
ps -eo pid,args | grep -E "bash temp/(queue_|run_|watchdog)|train_gpu" | grep -v grep | cut -c1-110
