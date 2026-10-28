#!/bin/bash
# n85_prf_sweep.sh -- 2026-10-28 night (Neuro 85): pattern completion (prf) and learned-free variants on the shared gated
# config, both benchmarks (multi-hop / multi-session are where Plasmem trails). CPU (GPU 0 hosts bge only). Ends DONE.
cd /home/glenn/projects/neuro || exit 1
C=g_sem=1,graded=1,qrate=1,signed=1,n_fly=65536,fly_k=4096,g_t=1,nb=0.5,nb_mode=1,nb_len=800
for b in lme locomo; do
  timeout 7200 bash temp/n85_bench.sh 0 n85_prf_sweep.log $b psy:$C psy:$C,prf=3,beta=0.2 psy:$C,prf=3,beta=0.4 psy:$C,prf=5,beta=0.3 psy:$C,g_sem=1.3 psy:$C,g_sem=0.8 psy:$C,nb=0.7
done
echo DONE >> logs/n85_prf_sweep.log
