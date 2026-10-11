#!/bin/bash
# hold_split.sh -- 2026-10-11 (Neuro 55): hold r26y so the 5090 chain exits after r26x, and start the aux-splits
# sanity driver (temp/queue_5090_split.sh), which restores the list and re-runs the chain.
cd /home/glenn/projects/neuro
pgrep -f "^bash temp/queue_5090_split.sh" > /dev/null && { echo "driver already running"; exit 0; }
bash -n temp/queue_5090_split.sh || exit 1
sed -i -E 's/^(r26y_)/#HOLD \1/' temp/queue_5090_n54.list
setsid -f bash temp/queue_5090_split.sh < /dev/null > /dev/null 2>&1
sleep 1; tail -n 3 temp/queue_5090_n54.list; pgrep -af "^bash temp/queue_5090" | cut -c1-90
