#!/bin/bash
# alert.sh -- 2026-10-09 (Neuro 54). One place for alerts: appends "[LEVEL mm-dd HH:MM:SS] msg" to logs/alerts.log
# (the session waiter temp/wait_alert.sh and Glenn read it) and, for ALERT / DONE, shows a Windows desktop toast.
# Levels: ALERT = needs a human or Claude | WARN = handled automatically or worth a look | DONE = result ready | INFO.
# Usage: bash temp/alert.sh LEVEL "message"
cd /home/glenn/projects/neuro
lv=$1; shift
echo "[$lv $(date '+%m-%d %H:%M:%S')] $*" >> logs/alerts.log
case $lv in ALERT|DONE)
  timeout 20 powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w temp/toast.ps1)" "neuro $lv" "$*" \
    > /dev/null 2>&1 < /dev/null ;;
esac
exit 0
