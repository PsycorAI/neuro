#!/bin/bash
# n86_wait_done.sh -- 2026-10-28 (Neuro 86). Session waiter: block until logs/$1 has a DONE line or $2 minutes pass
# (default 110 < the 2 h background limit), then print its tail. Run with run_in_background.
cd /home/glenn/projects/neuro
end=$(( $(date +%s) + ${2:-110} * 60 ))
until grep -q '^DONE' "logs/$1" 2>/dev/null || [ "$(date +%s)" -ge "$end" ]; do sleep 30; done
date; grep -q '^DONE' "logs/$1" || echo "TIMEOUT: no DONE in logs/$1"
tail -n 25 "logs/$1" | cut -c1-200
exit 0
