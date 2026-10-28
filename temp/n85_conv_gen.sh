#!/bin/bash
# n85_conv_gen.sh -- 2026-10-27 (Neuro 85, Glenn OK overnight): conversation-style pairs from SODA (CC BY 4.0) on both GPUs
# (5090: conv0, 5080: conv1 at <= ~13.5 GB), 4 h cap each (resumable), then stage 2 tuning from s1a on all pairs with hard
# negatives -> enc_s2c, then the full benchmark. Ends with DONE in logs/n85_conv_gen.log.
cd /home/glenn/projects/neuro || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
(QA_BUDGET=16000 timeout 14400 bash temp/n85_py.sh 0 n85_qgen_conv0.log qgen.py conv0 150000) &
(QA_BUDGET=8000 timeout 14400 bash temp/n85_py.sh 1 n85_qgen_conv1.log qgen.py conv1 100000) &
wait
echo "gen done" >> logs/n85_conv_gen.log
wc -l data/psymem_enc/pairs_*.jsonl >> logs/n85_conv_gen.log
(cd psymem && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 timeout 14400 /home/glenn/projects/bdh/venv/bin/python enc_tune.py s1a s2c 5 >> ../logs/n85_enc_tune.log 2>&1; echo "exit $? (s2c)" >> ../logs/n85_enc_tune.log)
C=g_sem=1,graded=1,qrate=1,signed=1,n_fly=65536,fly_k=4096,g_t=1
for b in lme locomo; do
  timeout 5400 bash temp/n85_bench.sh 0 n85_s2c_full.log $b psy:$C,nb=0.5 psy:$C,enc=s2a psy:$C,nb=0.5,enc=s2a psy:$C,enc=s2c psy:$C,nb=0.5,enc=s2c
done
echo DONE >> logs/n85_conv_gen.log
