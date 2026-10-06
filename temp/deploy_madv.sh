#!/bin/bash
# deploy_madv.sh -- 2026-09-30 (Glenn OK): deploy temp/stage_madv/data_stream.py (MADV_RANDOM for the independent-windows
# train path; byte-identical batches, temp/test_madv.py) into src/ atomically. Aborts if src changed since staging.
set -e
cd /home/glenn/projects/neuro
have=$(md5sum src/data_stream.py | cut -d' ' -f1)
[ "$have" = "778cfc6044ec6743a2c385b99ff13999" ] || { echo "ABORT: src/data_stream.py changed since staging ($have)"; exit 1; }
mkdir -p temp/predeploy_madv && cp -p src/data_stream.py temp/predeploy_madv/data_stream.py
cp temp/stage_madv/data_stream.py src/data_stream.py.new && mv src/data_stream.py.new src/data_stream.py
md5sum src/data_stream.py temp/stage_madv/data_stream.py
/home/glenn/projects/bdh/venv/bin/python temp/test_madv.py 2>&1 | head -1
echo "DEPLOYED (backup in temp/predeploy_madv)"
