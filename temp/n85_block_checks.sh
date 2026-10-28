#!/bin/bash
# n85_block_checks.sh -- 2026-10-28 (Neuro 85): after the block-storage core change: (1) old-vs-new parity test
# (logs/n85_core_parity.log, expect "PARITY OK ... worst abs diff 0"); (2) benchmark parity, gated config, on CPU-bound
# bench (logs/n85_block_parity.log; expect LME r@10 90.2, LoCoMo 76.7 as before). Ends DONE in logs/n85_block_parity.log.
cd /home/glenn/projects/neuro || exit 1
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 timeout 1500 /home/glenn/projects/bdh/venv/bin/python temp/n85_core_parity.py > logs/n85_core_parity.log 2>&1
C=g_sem=1,graded=1,qrate=1,signed=1,n_fly=65536,fly_k=4096,g_t=1,nb=0.5,nb_mode=1,nb_len=800
for b in lme locomo; do timeout 5400 bash temp/n85_bench.sh 1 n85_block_parity.log $b psy:$C; done
echo DONE >> logs/n85_block_parity.log
