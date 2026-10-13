#!/bin/bash
# Neuro 58, 10-12 13:55: temp/n58_revive.py ran the WSL VM out of RAM (it kept 8 GB of logits beside a 10 GB
# checkpoint) and every detached home process was killed. Restart them, each only if it is not running:
# alert supervisor, cloud watcher, logs-only pull, Drive pull of the newest checkpoint, probe B (resumes at 46,700)
# with its follower, the watchdog holder.
cd /home/glenn/projects/neuro || exit 1
. temp/n56_host.env
up(){ pgrep -f "$1" > /dev/null; }
up '^bash temp/alert_watch.sh' || setsid -f bash temp/alert_watch.sh < /dev/null > /dev/null 2>&1
up '^bash cloud/cloud_watch.sh' || MIN_TPS=13000 STALL_MIN=45 setsid -f bash cloud/cloud_watch.sh $H $P < /dev/null > /dev/null 2>&1
up '^bash temp/n58_pull_logs.sh' || setsid -f bash temp/n58_pull_logs.sh < /dev/null > /dev/null 2>&1
new=$(rclone lsf gdrive:neuro_v4_ckpts/v4_cloud/ 2>/dev/null | grep -E '^step_[0-9]+\.pt$' | sort | tail -n 1)
echo "newest on Drive: $new"
up '^bash temp/n57_drive_pull.sh' || { [ -n "$new" ] && setsid -f bash temp/n57_drive_pull.sh v4_cloud $new < /dev/null > /dev/null 2>&1; }
up '^bash temp/n58_probe.sh' || setsid -f bash temp/n58_probe.sh probe_v4_nodecay < /dev/null > /dev/null 2>&1
up '^bash temp/n58_probe_follow.sh' || setsid -f bash temp/n58_probe_follow.sh probe_v4_nodecay < /dev/null > /dev/null 2>&1
sleep 3
up '^bash temp/queue_5090_n58hold.sh' || setsid -f bash temp/queue_5090_n58hold.sh < /dev/null > /dev/null 2>&1
sleep 2
up '^bash temp/watchdog_tps.sh' || WD_SKIP='^v4_smoke1b|^screen_prof_|^probe_' MIN_TPS=2500 STALL_MIN=40 setsid -f bash temp/watchdog_tps.sh < /dev/null > /dev/null 2>&1
sleep 5
pgrep -af '^bash (temp|cloud)/' | cut -c1-90
free -g | sed -n 2p
