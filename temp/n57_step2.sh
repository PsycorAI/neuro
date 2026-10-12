#!/bin/bash
# Neuro 57 build step 2 (plan D1): add the say sentences, run the protocol v2 test on CPU.
cd /home/glenn/projects/neuro || exit 1
PY=/home/glenn/projects/bdh/venv/bin/python
nice $PY temp/n57_add_say.py || exit 1
CUDA_VISIBLE_DEVICES= nice $PY temp/test_tool_proto2.py 2>&1 | tail -n 12
