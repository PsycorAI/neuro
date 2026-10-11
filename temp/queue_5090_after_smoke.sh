#!/bin/bash
# queue_5090_after_smoke.sh -- 2026-10-10 (Neuro 54). After queue_5090_smoke1b.sh (1B smoke -> r26u) has exited, run
# the n54 chain again on whatever is new in temp/queue_5090_n54.list (r26v = the real early-start test).
# Launch: setsid -f bash temp/queue_5090_after_smoke.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro
while pgrep -f "^bash temp/queue_5090_smoke1b.sh" > /dev/null; do sleep 60; done
exec bash temp/queue_5090_n54.sh
