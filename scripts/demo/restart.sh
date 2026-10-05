#!/bin/bash
# restart.sh [run] [device] -- 2026-10-04: (re)start the demo server in the background (log: logs/demo_server.log).
# The [s] in the pattern stops pgrep matching this script's own command line.
cd /home/glenn/projects/neuro
RUN=${1:-screen_r22a_x1_s0_30m}; DEV=${2:-cpu}
for p in $(pgrep -f '[s]cripts/demo/server.py'); do kill $p; done
sleep 2
if [ "$DEV" = cpu ]; then export CUDA_VISIBLE_DEVICES=; fi
setsid -f /home/glenn/projects/bdh/venv/bin/python scripts/demo/server.py --run "$RUN" --device "$DEV" \
    > logs/demo_server.log 2>&1 < /dev/null
for i in $(seq 1 60); do grep -q 'http://localhost' logs/demo_server.log && break; sleep 2; done
tail -2 logs/demo_server.log
