#!/bin/bash
# n85_after_conv.sh -- 2026-10-28 night (Neuro 85): after the conv chain (n85_conv_gen.sh DONE), full benchmark of the
# shared gated config (nb_mode 1, nb_len 800) with each front end: bge, spiking s2a, spiking s2c. 5090. Ends DONE.
cd /home/glenn/projects/neuro || exit 1
timeout 21600 bash -c 'until grep -q DONE logs/n85_conv_gen.log; do sleep 60; done'
C=g_sem=1,graded=1,qrate=1,signed=1,n_fly=65536,fly_k=4096,g_t=1,nb=0.5,nb_mode=1,nb_len=800
for b in lme locomo; do
  timeout 5400 bash temp/n85_bench.sh 0 n85_front_gated.log $b hybrid_cc_nbg psy:$C psy:$C,enc=s2a psy:$C,enc=s2c
done
echo DONE >> logs/n85_front_gated.log
