#!/bin/bash
# Neuro 57: read-only. The full eval blocks of the v4 cloud run (everything the trainer prints at an eval), and the
# early eval lines of the 13L RC3 reference runs at home, to judge memory delta / spike rate early in training.
cd /home/glenn/projects/neuro || exit 1
. temp/n56_host.env
echo "== v4_cloud eval blocks (instance)"
$SSH root@$H 'cd /workspace/neuro; grep -n -E "^step (3000|6000|9000) |\[eval\]|\[valppl\]|\[beta|\[spike|\[wnorm|\[alpha|\[mqar|\[gate|\[homeo|\[ffn|recall|WATCHDOG" logs/v4_cloud.log | cut -c1-330 | tail -n 60; echo; grep -E "^step" logs/v4_cloud.log | tail -n 1 | cut -c1-200' 2>/dev/null
echo
echo "== home reference logs"
ls logs | grep -E "^r26n|^r26p|v4_cloud|reh|smoke1b|1b_" | head -n 30
true
