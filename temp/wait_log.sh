#!/bin/bash
# wait_log.sh -- 2026-10-05 (10-08: prints the matching NEW lines + context; exit 0 always). Block until a NEW line of
# logs/gap_queue.log (appended after this starts) matches the regex $1, or $2 minutes pass (default 110 < the 2 h
# background limit); then print the new matching block. One-shot background waiter. Never self-matches (no pgrep).
# Usage: bash temp/wait_log.sh '<regex>' [max_minutes] [context_lines]
cd /home/glenn/projects/neuro
n0=$(wc -l < logs/gap_queue.log)
end=$(( $(date +%s) + ${2:-110} * 60 ))
until tail -n +$((n0 + 1)) logs/gap_queue.log | grep -qE "$1" || [ "$(date +%s)" -ge "$end" ]; do sleep 60; done
date
if tail -n +$((n0 + 1)) logs/gap_queue.log | grep -qE "$1"; then
  tail -n +$((n0 + 1)) logs/gap_queue.log | grep -E -A "${3:-10}" "$1" | cut -c1-420
else
  echo "TIMEOUT after ${2:-110} min without a new line matching: $1"; tail -n 3 logs/gap_queue.log | cut -c1-200
fi
exit 0
