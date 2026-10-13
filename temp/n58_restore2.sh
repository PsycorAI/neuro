#!/bin/bash
# Neuro 58, 10-12 14:00: second restore. The first one (n58_restore.sh) started the probe trainer and its eval follower
# together = two 10 GB checkpoint loads at once on a 24 GB VM, and the distro restarted again at 13:57.
# Now: light watchers + the Drive pull + the trainer ONLY (no follower; evals of its checkpoints run by hand, alone).
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
sleep 3
up '^bash temp/queue_5090_n58hold.sh' || setsid -f bash temp/queue_5090_n58hold.sh < /dev/null > /dev/null 2>&1
sleep 2
up '^bash temp/watchdog_tps.sh' || WD_SKIP='^v4_smoke1b|^screen_prof_|^probe_' MIN_TPS=2500 STALL_MIN=40 setsid -f bash temp/watchdog_tps.sh < /dev/null > /dev/null 2>&1
sleep 5
pgrep -af '^bash (temp|cloud)/' | cut -c1-90
free -g | sed -n 2p
