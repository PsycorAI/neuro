#!/bin/bash
# kill_stuck_haystack.sh -- 2026-10-11 (Neuro 55): temp/haystack_recall_eval.py of screen_p26zg has been running 1 h 50
# on the 5080 at 100% util / 15.9 GB (WDDM spill; normal: under a minute). It has no time limit in run_5080_sched.sh
# and blocks the 5080 chain (p26zh, p26zi, p26zj). SIGTERM that one eval; the chain then goes on to the r25 evals and
# the next screen. Only this eval is touched (exact command-line match), no trainer.
p=$(pgrep -f "temp/haystack_recall_eval.py screen_p26zg_7L_RC3_gap20_x20=")
[ -n "$p" ] || { echo "not running"; exit 0; }
echo "stuck eval pid $p, up $(ps -o etimes= -p $p | tr -d ' ') s"
kill -TERM $p; sleep 5
pgrep -f "temp/haystack_recall_eval.py screen_p26zg_7L_RC3_gap20_x20=" > /dev/null && { echo "still alive -> SIGKILL"; kill -KILL $p; }
sleep 2; echo "[n55 $(date '+%m-%d %H:%M:%S')] [screen_p26zg_7L_RC3_gap20_x20] haystack_recall_eval killed after a 1 h 50 WDDM spill (no result line; to rerun on the 5090)" >> /home/glenn/projects/neuro/logs/gap_queue.log
ps -eo pid,etimes,args | grep -E 'eval' | grep -v grep | cut -c1-160
