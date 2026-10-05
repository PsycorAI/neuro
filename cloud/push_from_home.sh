#!/bin/bash
# push_from_home.sh <ssh-host> <ssh-port> -- run in WSL at home (2026-10-04 draft). Copies code, configs, the vocab map
# and the uint16 v4 corpus (46 GB, LUT already applied -> data_prelut: true) to /workspace on the Vast instance.
# rsync is resumable: re-run it if the upload is interrupted.
set -e
H=${1:?ssh host}; P=${2:?ssh port}
SSH="ssh -p $P -o StrictHostKeyChecking=accept-new"
cd /home/glenn/projects/neuro
$SSH root@$H "mkdir -p /workspace/neuro/data /workspace/data/tokenized_v4 /workspace/neuro/logs /workspace/neuro/models/checkpoints"
rsync -az --info=progress2 -e "$SSH" src scripts cloud root@$H:/workspace/neuro/
$SSH root@$H "mkdir -p /workspace/neuro/temp"
rsync -az -e "$SSH" temp/profile_train.py temp/haystack_recall_eval.py root@$H:/workspace/neuro/temp/
rsync -az -e "$SSH" configs/v4_cloud.yaml root@$H:/workspace/neuro/configs/
rsync -az -e "$SSH" data/vocab_map_32768.npy root@$H:/workspace/neuro/data/
rsync -a --partial --info=progress2 -e "$SSH" \
    /home/glenn/projects/bdh/data/tokenized_v4/train_lut32768_u16.bin \
    /home/glenn/projects/bdh/data/tokenized_v4/train_lut32768_u16.json root@$H:/workspace/data/tokenized_v4/
echo "[push] done"
