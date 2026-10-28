#!/bin/bash
# n85_nbgate.sh -- 2026-10-27 night (Neuro 85): gated S<->S neighbour links (core.activate nb_mode) on both benchmarks,
# to find one config that wins LoCoMo and LME. bge front end, cached embeddings (CPU work; GPU 0 only hosts bge).
cd /home/glenn/projects/neuro || exit 1
C=g_sem=1,graded=1,qrate=1,signed=1,n_fly=65536,fly_k=4096,g_t=1
for b in lme locomo; do
  timeout 7200 bash temp/n85_bench.sh 0 n85_nbgate.log $b psy:$C psy:$C,nb=0.5 psy:$C,nb=0.5,nb_mode=1,nb_len=300 psy:$C,nb=0.5,nb_mode=1,nb_len=800 psy:$C,nb=0.5,nb_mode=2 psy:$C,nb=0.5,nb_mode=3,nb_len=300 psy:$C,nb=0.5,nb_mode=3,nb_len=800
done
echo DONE >> logs/n85_nbgate.log
