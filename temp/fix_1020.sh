#!/bin/bash
# fix_1020.sh -- watchdog also covers queue_spk_* scripts; commit the sweep.
cd /home/glenn/projects/neuro
for p in $(pgrep -f "bash temp/watchdog_tps.sh"); do kill $p; done
grep -q 'queue_spk_' temp/watchdog_tps.sh || sed -i 's/\[q\]ueue_50\[89\]0_|/[q]ueue_50[89]0_|[q]ueue_spk_|/' temp/watchdog_tps.sh
grep -n '^while pgrep' temp/watchdog_tps.sh
bash -n temp/watchdog_tps.sh && setsid -f bash temp/watchdog_tps.sh > /dev/null 2>&1
git add configs/r25*.yaml configs/screen_p25*.yaml logs/gap_queue.log
git add -f temp/queue_spk_*.sh temp/watchdog_tps.sh temp/mk_r25_configs.py temp/fix_1020.sh
git commit -qm 'spike sweep queued: r25t L8 80M; p25s2/s3/s4/p25t

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'
git log --oneline -1
pgrep -fc "bash temp/watchdog_tps.sh"
