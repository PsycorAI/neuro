#!/bin/bash
# test_wait_free.sh -- 2026-09-30: dry-test the wait_free() added to run_5080_sched.sh (no training, no lock, own log)
GPU=1; LOG=/tmp/test_wait_free.log; : > $LOG
say(){ echo "[test $(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
eval "$(sed -n '/^wait_free(){/,/^}/p' /home/glenn/projects/neuro/temp/run_5080_sched.sh.new)"
FREE_UTIL=100 FREE_MIN=1; wait_free "pass-case"; echo "pass-case returned (expected immediately)"
FREE_UTIL=-1 FREE_MIN=1; timeout 5 bash -c "$(declare -f say wait_free); GPU=1 LOG=$LOG FREE_UTIL=-1 FREE_MIN=1 wait_free busy-case"; echo "busy-case rc=$? (expected 124 = still waiting)"
cat $LOG
