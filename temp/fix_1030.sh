#!/bin/bash
# fix_1030.sh -- 2026-10-05: 5080 order after p25e -> p25b (chunk-only diagnostic) first, then d, c, h.
cd /home/glenn/projects/neuro
for p in $(pgrep -f "bash temp/queue_5080_r25.sh"); do kill $p; done
sleep 1
sed -i 's/^for r in screen_p25g_6L_chunk512_x_fix screen_p25i_6L_chunk512_x_read screen_p25e_6L_chunk512_x_H3 \\$/for r in screen_p25e_6L_chunk512_x_H3 screen_p25b_6L_chunk512 \\/; s/^         screen_p25h_6L_chunk512_x_nl_fix screen_p25d_6L_chunk512_T2048_x screen_p25b_6L_chunk512 screen_p25c_6L_chunk256_x; do$/         screen_p25d_6L_chunk512_T2048_x screen_p25c_6L_chunk256_x screen_p25h_6L_chunk512_x_nl_fix; do/' temp/queue_5080_r25.sh
grep -A1 '^for r in' temp/queue_5080_r25.sh
bash -n temp/queue_5080_r25.sh && setsid -f bash temp/queue_5080_r25.sh > /dev/null 2>&1
echo "[s1 $(date '+%m-%d %H:%M:%S')] 5080 chain reordered: p25e (running) -> p25b (chunk-only diagnostic) -> p25d -> p25c -> p25h" >> logs/gap_queue.log
