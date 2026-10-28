#!/bin/bash
# n85_ramguard.sh -- 2026-10-28 (Neuro 85): kill my CPU experiments (n85_scale / bench) before the VM runs out of RAM,
# so an OOM never starts the WSL kill loop again. Never touches qgen / enc_tune / the demo server. Runs until killed.
while true; do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$avail" -lt 3 ]; then
    pkill -f 'temp/n85_scale.py' && echo "$(date) ramguard: killed n85_scale (avail ${avail} GB)" >> /home/glenn/projects/neuro/logs/n85_ramguard.log
  fi
  sleep 5
done
