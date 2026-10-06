#!/bin/bash
# fix_0945.sh -- 2026-10-06: stop r25q (slow NL test, superseded by the spiking priority), start the spike conversions:
# 5090 r25s (from r25o) now; 5080 p25s (from p25o) after the p25pq queue, then p25r.
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log
for p in $(pgrep -f "bash temp/queue_5090_r25q.sh"); do kill $p; done
for p in $(pgrep -f "configs/r25q_13L_H6only_nl_150m.yaml"); do kill -TERM $p; done
for p in $(pgrep -f "bash temp/queue_5080_p25r.sh"); do kill $p; done
sleep 5
/home/glenn/projects/bdh/venv/bin/python temp/mk_r25_configs.py > /dev/null
echo "[q0 $(date '+%m-%d %H:%M:%S')] r25q STOPPED at 34M tok (10k tok/s; NL aux never helped) -> energy plan: spike conversions r25s (5090) / p25s (5080)" >> $LOG
setsid -f bash temp/queue_spk_conv.sh 0 r25s_13L_spk_conv_150m models/checkpoints/r25o_13L_H6only_150m/step_0012208.pt > /dev/null 2>&1
setsid -f bash temp/queue_spk_conv.sh 1 screen_p25s_7L_spk_conv models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt "bash temp/queue_5080_p25pq.sh" > /dev/null 2>&1
sleep 1
sed 's/while pgrep -f "bash temp\/queue_5080_p25pq.sh"/while pgrep -f "bash temp\/queue_spk_conv.sh 1 "/' temp/queue_5080_p25r.sh > temp/queue_5080_p25r2.sh
setsid -f bash temp/queue_5080_p25r2.sh > /dev/null 2>&1
sleep 2; ps -eo pid,args | grep -E "bash temp/(queue_|run_|watchdog)|train_gpu" | grep -v grep | cut -c1-110
