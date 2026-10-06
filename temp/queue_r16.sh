#!/bin/bash
# queue_r16.sh -- 2026-10-02 (Glenn OK). Round 16 on the 5080 behind the r15 scheduler (blocks on the GPU-1 lock).
# Order = most decision-relevant first. WATCH: tail -f /home/glenn/projects/neuro/logs/gap_queue.log
cd /home/glenn/projects/neuro
GPU=1 setsid -f bash temp/run_5080_sched.sh \
  screen_r16_c0_s0_30m screen_r16_l3_gelu_s0_30m screen_r16_x2_dsnlif_s0_30m screen_r16_c0_s1_30m \
  screen_r16_t4_tfmr4_fair_s0_30m screen_r16_l1_oldhash_s0_30m screen_r16_l2_nogate_s0_30m \
  screen_r16_l4_lev4_s0_30m screen_r16_l5_outnorm_s0_30m screen_r16_x1_lategap_s0_30m \
  > logs/queue_r16.out 2>&1 < /dev/null
sleep 3
ps -eo pid,args | grep "[r]un_5080_sched" | cut -c1-160
tail -n 2 logs/gap_queue.log
