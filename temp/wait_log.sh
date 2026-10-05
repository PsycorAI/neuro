#!/bin/bash
# wait_log.sh -- 2026-10-05. Block until logs/gap_queue.log matches the regex $1 or $2 minutes pass, then print the
# recent result lines. For one-shot background waiters (one notification instead of polling).
# Usage: bash temp/wait_log.sh '<regex>' <max_minutes>
cd /home/glenn/projects/neuro
end=$(( $(date +%s) + ${2:-110} * 60 ))
until grep -qE "$1" logs/gap_queue.log || [ "$(date +%s)" -ge "$end" ]; do sleep 60; done
date
grep -E 'exit=|r25-evals|haystack|recall|\[wd|not DONE|SKIPPED' logs/gap_queue.log | tail -${3:-16} | cut -c1-420
