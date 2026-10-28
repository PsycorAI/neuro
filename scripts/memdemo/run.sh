#!/bin/bash
# scripts/memdemo/run.sh -- 2026-10-27 (Neuro 85): start the Plasmem demo (CPU only) in the background on port 8766.
# Log: logs/memdemo/server.log. Stop: pkill -f "^/home/glenn/projects/bdh/venv/bin/python scripts/memdemo/server.py"
cd /home/glenn/projects/neuro || exit 1
mkdir -p logs/memdemo
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 setsid -f /home/glenn/projects/bdh/venv/bin/python scripts/memdemo/server.py "$@" > logs/memdemo/server.log 2>&1 < /dev/null
