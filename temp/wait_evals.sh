#!/bin/bash
# wait_evals.sh LOG N [TIMEOUT_S] -- wait until LOG has >= N "[eval] val_ppl" lines, then print them.
LOG=$1; N=$2; TO=${3:-590}; end=$((SECONDS + TO))
while [ "$(grep -c 'eval\] val_ppl' "$LOG")" -lt "$N" ] && [ $SECONDS -lt $end ]; do sleep 20; done
date; grep -E 'eval\]|COMPLETE|STOPPED|Traceback' "$LOG"
