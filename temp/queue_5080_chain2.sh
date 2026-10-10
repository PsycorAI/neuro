#!/bin/bash
# queue_5080_chain.sh -- 2026-10-07 15:45 (Neuro 53; Glenn: 5080 free through 18:00 10-08). After p26y (queue_sc.sh):
# p26z (7L + HOLA) -> p26za (7L + HOLA + stochastic chunk), each with r25 evals + E1-E3.
cd /home/glenn/projects/neuro
true
UNTIL="2026-10-08 17:55" bash temp/queue_5080_p26.sh screen_p26z_7L_hola64 screen_p26za_7L_hola64_rand128
echo "[n53 $(date '+%m-%d %H:%M:%S')] 5080 chain (p26z, p26za) DONE" >> logs/gap_queue.log
