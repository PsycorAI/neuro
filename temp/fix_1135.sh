#!/bin/bash
cd /home/glenn/projects/neuro
for p in $(pgrep -f "^bash temp/queue_5080_p26x.sh$"); do echo "TERM $p"; kill -TERM $p; done
sleep 1
sed -i "s/pgrep -f \"bash temp\/queue_5080_(site|p26)\"/pgrep -f \"bash temp\/queue_5080_(site|p26)\\.sh\"/" temp/queue_5080_p26x.sh
grep -n "while pgrep" temp/queue_5080_p26x.sh
bash -n temp/queue_5080_p26x.sh && echo SYNTAX_OK
