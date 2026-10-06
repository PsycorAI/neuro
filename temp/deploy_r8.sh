#!/bin/bash
# deploy_r8.sh -- 2026-09-29 (Glenn OK): deploy temp/stage_r8 (stateful path fix + attn_window + SequentialStream.skip)
# into src/ atomically. Aborts if any live file changed since it was staged (md5 taken at staging time).
set -e
cd /home/glenn/projects/neuro
declare -A WANT=([train_gpu.py]=8479d8b592fb4180644e44d181aab09b [hybrid_attn.py]=4b6b1043522353ed2884004f68566ebc [data_stream.py]=b0e7dbe6a329aee9223edc76ba28c475)
for f in "${!WANT[@]}"; do
  have=$(md5sum src/$f | cut -d' ' -f1)
  [ "$have" = "${WANT[$f]}" ] || { echo "ABORT: src/$f changed since staging ($have)"; exit 1; }
done
mkdir -p temp/predeploy_r8
for f in "${!WANT[@]}"; do cp -p src/$f temp/predeploy_r8/$f; cp temp/stage_r8/$f src/$f.new; done
for f in "${!WANT[@]}"; do mv src/$f.new src/$f; done
md5sum src/train_gpu.py src/hybrid_attn.py src/data_stream.py temp/stage_r8/train_gpu.py temp/stage_r8/hybrid_attn.py temp/stage_r8/data_stream.py
echo "DEPLOYED (backup of previous files in temp/predeploy_r8)"
