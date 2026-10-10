#!/bin/bash
cd /home/glenn/projects/neuro
for p in $(pgrep -f "^bash temp/queue_5080_site.sh$"); do echo "TERM $p"; kill -TERM $p; done
sleep 1
sed -i "s/while pgrep -f \"bash temp\/queue_5080_p26\" /while pgrep -f \"bash temp\/queue_5080_p26.sh\" /" temp/queue_5080_site.sh
grep -n "while pgrep" temp/queue_5080_site.sh
bash -n temp/queue_5080_site.sh && echo SYNTAX_OK
setsid -f bash temp/queue_5080_site.sh > /dev/null 2>&1
sleep 1; pgrep -af "queue_5080"
