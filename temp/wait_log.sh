#!/bin/bash
# wait_log.sh -- 2026-10-05. Block until a NEW line of logs/gap_queue.log (appended after this starts) matches the
# regex $1, or $2 minutes pass; then print the recent result lines. One-shot background waiter.
# Usage: bash temp/wait_log.sh '<regex>' <max_minutes> [<n_lines>]
cd /home/glenn/projects/neuro
n0=$(wc -l < logs/gap_queue.log)
end=$(( $(date +%s) + ${2:-110} * 60 ))
until tail -n +$((n0 + 1)) logs/gap_queue.log | grep -qE "$1" || [ "$(date +%s)" -ge "$end" ]; do sleep 60; done
date
grep -E 'exit=|r25-evals|haystack|recall|\[wd|not DONE|SKIPPED' logs/gap_queue.log | tail -${3:-16} | cut -c1-420
