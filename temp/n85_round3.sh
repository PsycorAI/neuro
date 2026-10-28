#!/bin/bash
# n85_round3.sh -- 2026-10-28 night (Neuro 85): after the gated front-end benchmark (n85_front_gated.log DONE):
# 5090: two more stage-2 variants (s2d = s1a x 10 epochs; s2e = from s1b, 5 epochs) + their gated benchmarks;
# 5080: keep generating SODA conversation pairs (conv1, resumable) until 06:40 for the next round. Ends DONE.
cd /home/glenn/projects/neuro || exit 1
timeout 25000 bash -c 'until grep -q DONE logs/n85_front_gated.log 2>/dev/null; do sleep 60; done'
left=$(( $(date -d 06:40 +%s) - $(date +%s) ))
[ $left -gt 600 ] && (QA_BUDGET=8000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout $left bash temp/n85_py.sh 1 n85_qgen_conv1.log qgen.py conv1 220000) &
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/glenn/projects/bdh/venv/bin/python
(cd psymem && timeout 5400 $PY enc_tune.py s1a s2d 10 >> ../logs/n85_enc_tune.log 2>&1; echo "exit $? (s2d)" >> ../logs/n85_enc_tune.log)
(cd psymem && timeout 5400 $PY enc_tune.py s1b s2e 5 >> ../logs/n85_enc_tune.log 2>&1; echo "exit $? (s2e)" >> ../logs/n85_enc_tune.log)
C=g_sem=1,graded=1,qrate=1,signed=1,n_fly=65536,fly_k=4096,g_t=1,nb=0.5,nb_mode=1,nb_len=800
for b in lme locomo; do
  timeout 5400 bash temp/n85_bench.sh 0 n85_round3.log $b psy:$C,enc=s2d psy:$C,enc=s2e
done
echo DONE >> logs/n85_round3.log
