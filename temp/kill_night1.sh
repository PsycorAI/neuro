#!/bin/bash
# kill the superseded queue_5090_night.sh (sleeping in its 00:30 wait; holds 5090_seq.lock)
P=$(pgrep -f '[q]ueue_5090_night\.sh')
echo "old night queue PID(s): $P"
[ -n "$P" ] && kill $P
sleep 3
pgrep -af '[q]ueue_5090_night' || true
