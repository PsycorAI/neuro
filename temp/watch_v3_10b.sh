#!/bin/bash
# Exit (one notification) when flagship_v3 reaches 10B tokens, the process exits, or the log shows an error.
LOG=/home/glenn/projects/neuro/logs/flagship_v3_363m.log
while true; do
  last=$(grep -E '^step [0-9]+ \|' "$LOG" | tail -n 1)
  tok=$(echo "$last" | sed -E 's/.*\| ([0-9.]+)M tok.*/\1/' | cut -d. -f1)
  if [ -n "$tok" ] && [ "$tok" -ge 10000 ]; then echo "REACHED 10B: $last"; exit 0; fi
  if ! kill -0 569 2>/dev/null; then echo "PROCESS 569 EXITED. last: $last"; tail -n 5 "$LOG"; exit 0; fi
  if tail -n 50 "$LOG" | grep -qE 'Traceback|nan|NaN|OOM|out of memory|watchdog.*trip'; then echo "ERROR SIGNAL in log. last: $last"; tail -n 20 "$LOG"; exit 0; fi
  sleep 60
done
