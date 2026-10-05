#!/bin/bash
# fix_1300.sh -- 2026-10-05: gen r25i/r25j/p25l, queue r25i+j on the 5090, 5080 order p25b (running) -> p25l -> d -> c.
cd /home/glenn/projects/neuro
/home/glenn/projects/bdh/venv/bin/python temp/mk_r25_configs.py | grep -E 'r25i|r25j|p25l'
setsid -f bash temp/queue_5090_r25ij.sh > /dev/null 2>&1
for p in $(pgrep -f "bash temp/queue_5080_r25.sh"); do kill $p; done
sleep 1
sed -i 's/^for r in screen_p25e_6L_chunk512_x_H3 screen_p25b_6L_chunk512 \\$/for r in screen_p25b_6L_chunk512 screen_p25l_6L_chunk512_x_H3_nl \\/; s/^         screen_p25d_6L_chunk512_T2048_x screen_p25c_6L_chunk256_x screen_p25h_6L_chunk512_x_nl_fix; do$/         screen_p25d_6L_chunk512_T2048_x screen_p25c_6L_chunk256_x; do/' temp/queue_5080_r25.sh
grep -A1 '^for r in' temp/queue_5080_r25.sh
bash -n temp/queue_5080_r25.sh && setsid -f bash temp/queue_5080_r25.sh > /dev/null 2>&1
echo "[q0 $(date '+%m-%d %H:%M:%S')] p25e (6L H0+H3) cross-chunk haystack N8 ~89% / N32 ~80% (mem off chance), ppl +10%; r25h (12L late, xchunk 1.0 w 0.5) cross-chunk N8 59-79% / N32 ~30%, chunk1 40.47 (+4%), in-window haystack fell. Queued: 5090 r25i (balanced late phase) -> r25j (12L H0+H6 from scratch); 5080 p25b -> p25l (H3 + NL) -> p25d -> p25c" >> logs/gap_queue.log
sleep 2; ps -eo pid,args | grep -E "bash temp/(queue_|run_)|train_gpu" | grep -v grep | cut -c1-100
