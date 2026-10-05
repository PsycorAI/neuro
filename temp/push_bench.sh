#!/bin/bash
# push_bench.sh -- 2026-10-04: code + bench inputs only (no 46 GB corpus; Glenn's upload is 1.3 Mbps).
set -e
H=144.79.51.2; P=50980
SSH="ssh -p $P -o StrictHostKeyChecking=accept-new"
cd /home/glenn/projects/neuro
$SSH root@$H "mkdir -p /workspace/neuro/data /workspace/neuro/logs /workspace/neuro/models/checkpoints /workspace/neuro/temp /workspace/neuro/configs"
rsync -az --exclude __pycache__ -e "$SSH" src scripts cloud root@$H:/workspace/neuro/
rsync -az -e "$SSH" temp/profile_train.py temp/bench_slice_u16.bin root@$H:/workspace/neuro/temp/
rsync -az -e "$SSH" configs/v4_cloud_bench.yaml root@$H:/workspace/neuro/configs/
rsync -az -e "$SSH" data/vocab_map_32768.npy root@$H:/workspace/neuro/data/
echo "[push] done"
