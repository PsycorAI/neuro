#!/bin/bash
# Neuro 57: 13L SFT arm = protocol v2 + the second-pass (tool result -> reply) data, on the 5090.
cd /home/glenn/projects/neuro || exit 1
PY=/home/glenn/projects/bdh/venv/bin/python; PT=/home/glenn/projects/neuro/training_data/posttrain
$PY temp/n57_mk_toolresult_blend.py || exit 1
$PY temp/n57_mk_arm.py sft_v4_reh13_v2tr tool_proto=2 sft_tools_full_frac=0.3 \
  "sft_jsonl=[$PT/tools.jsonl, $PT/coldstart.jsonl, $PT/toolresult_blend.jsonl]" || exit 1
setsid -f bash temp/queue_5090_sftarm.sh sft_v4_reh13_v2tr 0 < /dev/null > /dev/null 2>&1
echo launched
