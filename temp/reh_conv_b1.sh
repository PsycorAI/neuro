#!/bin/bash
# reh_conv_b1.sh -- 2026-10-10 (Neuro 54). One-shot: regenerate the v4 configs (v4_reh_conv -> B1xGA8) and SIGTERM the
# rehearsal's conversion trainer so cloud/run_v4.sh resumes it with the new batch split. Kills only if the config changed.
cd /home/glenn/projects/neuro
/home/glenn/projects/bdh/venv/bin/python temp/mk_v4_cloud.py 2>&1 | grep -E '^\[OK\]|Error' | head -n 2
grep -qx 'batch_size: 1' configs/v4_reh_conv.yaml || { echo "config NOT changed: not restarting"; exit 1; }
git diff --stat -- configs/v4_cloud.yaml configs/v4_cloud_conv.yaml | tail -n 1
bash temp/alert.sh INFO "rehearsal conversion: planned restart at B1xGA8 (5080 spill) -- its exit=143 is expected"
pkill -TERM -f "src/train_gpu.py --config configs/v4_reh_conv.yaml"
sleep 100
tail -n 2 logs/run_v4.log
grep -E '^resumed|^arch=' logs/v4_reh_conv.log | tail -n 1 | cut -c1-150
grep -cE '^(batch_size: 1|grad_accum: 8)$' configs/v4_reh_conv.yaml
