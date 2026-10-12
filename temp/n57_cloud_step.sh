#!/bin/bash
# Neuro 57: read-only look at the v4 cloud run (latest step / eval / checkpoint lines) and at the home checkpoint copies.
cd /home/glenn/projects/neuro || exit 1
. temp/n56_host.env
echo "== instance"
$SSH root@$H 'cd /workspace/neuro; date -u "+%m-%d %H:%M:%S UTC"; grep -E "^step" logs/v4_cloud.log | tail -n 2 | cut -c1-300; grep -E "eval\]" logs/v4_cloud.log | tail -n 4 | cut -c1-320; grep -E "saved|WATCHDOG|Traceback|Error" logs/v4_cloud.log | tail -n 4 | cut -c1-220; ls -la models/checkpoints/v4_cloud/ | tail -n 6; tail -n 3 logs/run_v4.log | cut -c1-220' 2>/dev/null
echo "== home copies"
ls -la models/checkpoints/v4_cloud/ 2>/dev/null || echo "no home copy dir yet"
ls -la logs/cloud/ 2>/dev/null | tail -n 8
echo "== home rsync"
pgrep -af '^rsync' | cut -c1-170 || echo "no rsync running"
echo "== waiters"
pgrep -af '^bash temp/wait_alert.sh' || echo "no waiter"
true
