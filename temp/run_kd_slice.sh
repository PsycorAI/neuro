#!/bin/bash
# run_kd_slice.sh -- 2026-10-09 (Neuro 54). Builds the raw uint32 KD slice (temp/mk_kd_slice.py, 8 GB on G:) at low
# priority and reports through temp/alert.sh. Launch: setsid -f bash temp/run_kd_slice.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro
out=$(nice -n 19 ionice -c 3 /home/glenn/projects/bdh/venv/bin/python temp/mk_kd_slice.py 2>&1 | tail -1 | cut -c1-200)
case "$out" in "[OK]"*) bash temp/alert.sh DONE "KD slice built: $out" ;; *) bash temp/alert.sh ALERT "KD slice FAILED: $out" ;; esac
