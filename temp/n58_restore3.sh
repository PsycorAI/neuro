#!/bin/bash
# Neuro 58, 10-12 14:07, after Glenn's `wsl --shutdown`: restart the home side. Light watchers, the "next Drive
# checkpoint" puller, and the safety run (probe B, resumes at 46,700) as the ONLY checkpoint-loading job.
cd /home/glenn/projects/neuro || exit 1
. temp/n56_host.env
up(){ pgrep -f "$1" > /dev/null; }
up '^bash temp/alert_watch.sh' || setsid -f bash temp/alert_watch.sh < /dev/null > /dev/null 2>&1
up '^bash cloud/cloud_watch.sh' || MIN_TPS=13000 STALL_MIN=45 setsid -f bash cloud/cloud_watch.sh $H $P < /dev/null > /dev/null 2>&1
up '^bash temp/n58_pull_logs.sh' || setsid -f bash temp/n58_pull_logs.sh < /dev/null > /dev/null 2>&1
up '^bash temp/(n58_drive_next|n57_drive_pull).sh' || setsid -f bash temp/n58_drive_next.sh < /dev/null > /dev/null 2>&1
up '^bash temp/n58_probe.sh' || setsid -f bash temp/n58_probe.sh probe_v4_nodecay < /dev/null > /dev/null 2>&1
sleep 3
up '^bash temp/queue_5090_n58hold.sh' || setsid -f bash temp/queue_5090_n58hold.sh < /dev/null > /dev/null 2>&1
sleep 2
up '^bash temp/watchdog_tps.sh' || WD_SKIP='^v4_smoke1b|^screen_prof_|^probe_' MIN_TPS=2500 STALL_MIN=40 setsid -f bash temp/watchdog_tps.sh < /dev/null > /dev/null 2>&1
sleep 5
pgrep -af '^bash (temp|cloud)/' | cut -c1-90
free -g | sed -n 2p; systemctl status init.scope --no-pager 2>/dev/null | sed -n 4p
