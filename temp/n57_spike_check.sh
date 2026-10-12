#!/bin/bash
# Neuro 57: read-only. Spike rate / write gate / memory delta of the v4 cloud run over all evals; home watcher list.
cd /home/glenn/projects/neuro || exit 1
. temp/n56_host.env
$SSH root@$H 'cd /workspace/neuro; grep -E "^step [0-9]+000 |\[eval\]" logs/v4_cloud.log | paste - - 2>/dev/null | sed -E "s/ \| loss.*\[eval\]/ |/" | cut -c1-120 | awk "NR%2==1" | tail -n 16; echo; grep -E "\[beta\]|\[gates sat\]|\[alpha\]" logs/v4_cloud.log | grep -o -E "^ *\[[a-z ]+\]|blk6: [^|]*" | paste - - | tail -n 9; grep -E "^step" logs/v4_cloud.log | tail -n 1 | cut -c1-110; grep -c -i -E "nan|WATCHDOG" logs/v4_cloud.log' 2>/dev/null
echo "== home"; pgrep -af '^bash (cloud/cloud_watch|cloud/pull_ckpts|temp/alert_watch|temp/watchdog_tps)' | cut -c1-80
true
