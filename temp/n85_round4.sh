#!/bin/bash
# n85_round4.sh -- 2026-10-28 (Neuro 85): round 3 left s2d trained but unbenchmarked (s2e from s1b crashed again: "device
# not ready", VRAM; s1b dropped). 5080: gated benchmark of s2d now. 5090: s2f = s1a + all 530k pairs (354k SODA
# conversation pairs, conv0 + conv1 complete), 5 epochs, then its gated benchmark. Ends DONE in logs/n85_round4.log.
cd /home/glenn/projects/neuro || exit 1
C=g_sem=1,graded=1,qrate=1,signed=1,n_fly=65536,fly_k=4096,g_t=1,nb=0.5,nb_mode=1,nb_len=800
(for b in lme locomo; do timeout 5400 bash temp/n85_bench.sh 1 n85_round4.log $b psy:$C,enc=s2d; done) &
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/glenn/projects/bdh/venv/bin/python
(cd psymem && timeout 7200 $PY enc_tune.py s1a s2f 5 >> ../logs/n85_enc_tune.log 2>&1; echo "exit $? (s2f)" >> ../logs/n85_enc_tune.log)
wait
for b in lme locomo; do timeout 5400 bash temp/n85_bench.sh 0 n85_round4.log $b psy:$C,enc=s2f; done
echo DONE >> logs/n85_round4.log
