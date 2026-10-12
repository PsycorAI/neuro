#!/bin/bash
# Neuro 57: make + launch the 13L SFT arm with tool protocol v2 (plan D1) on the 5090.
cd /home/glenn/projects/neuro || exit 1
PY=/home/glenn/projects/bdh/venv/bin/python
$PY temp/n57_mk_arm.py sft_v4_reh13_v2 tool_proto=2 sft_tools_full_frac=0.3 || exit 1
bash -n temp/queue_5090_sftarm.sh || exit 1
setsid -f bash temp/queue_5090_sftarm.sh sft_v4_reh13_v2 0 < /dev/null > /dev/null 2>&1
sleep 90
pgrep -af "^bash temp/queue_5090_sftarm.sh" | cut -c1-100
tail -n 6 logs/sft_v4_reh13_v2.log | cut -c1-200
