#!/bin/bash
# n85_nbgate2.sh -- 2026-10-27 night (Neuro 85): confirm the winning gated-link config (nb 0.5, nb_mode 1, nb_len 800) over 3
# FlyHash seeds on both benchmarks, vs the ordinary baselines incl. hybrid_cc_nbg (the same length gate, for fairness).
cd /home/glenn/projects/neuro || exit 1
C=g_sem=1,graded=1,qrate=1,signed=1,n_fly=65536,fly_k=4096,g_t=1,nb=0.5,nb_mode=1,nb_len=800
for b in lme locomo; do
  timeout 7200 bash temp/n85_bench.sh 0 n85_nbgate2.log $b hybrid hybrid_cc hybrid_cc_nb hybrid_cc_nbg psy:$C psy:$C,seed=1 psy:$C,seed=2
done
echo DONE >> logs/n85_nbgate2.log
