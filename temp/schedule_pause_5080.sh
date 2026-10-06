#!/bin/bash
# Run pause_5080.sh at 18:50 today (Glenn needs the 5080 at ~19:00).
target=$(date -d "18:50" +%s); now=$(date +%s)
[ "$target" -gt "$now" ] && sleep $((target - now))
bash /home/glenn/projects/neuro/temp/pause_5080.sh
