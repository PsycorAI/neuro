#!/bin/bash
# deploy_spklog.sh -- 2026-09-30 (Glenn OK): [spike] log double-count fix + sparsity ramp flag (train_gpu.py) and the
# energy.py AC double-count fix. Aborts if either live file changed since staging.
set -e
cd /home/glenn/projects/neuro
[ "$(md5sum src/train_gpu.py | cut -d' ' -f1)" = "9d35434fa3eb8183ad47542da45be7cc" ] || { echo "ABORT: src/train_gpu.py changed"; exit 1; }
[ "$(md5sum src/energy.py | cut -d' ' -f1)" = "267923107d9826ebeae433a8ce1e9072" ] || { echo "ABORT: src/energy.py changed"; exit 1; }
mkdir -p temp/predeploy_spklog && cp -p src/train_gpu.py src/energy.py temp/predeploy_spklog/
for f in train_gpu.py energy.py; do cp temp/stage_spklog/$f src/$f.new; done
for f in train_gpu.py energy.py; do mv src/$f.new src/$f; done
md5sum src/train_gpu.py temp/stage_spklog/train_gpu.py src/energy.py temp/stage_spklog/energy.py
echo "DEPLOYED (backups in temp/predeploy_spklog)"
