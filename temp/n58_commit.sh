#!/bin/bash
# Neuro 58: commit Neuro 57's docs + build code (no logs, no data/demo, no push).
cd /home/glenn/projects/neuro || exit 1
set -e
git add docs/journey.html docs/to-do.html docs/HANDOFF_2026_10_12.md docs/HANDOFF_2026_10_12b.md docs/demo_v4_plan_2026_10_12.md
git add src/grammar.py src/tool_data.py src/train_sft.py training_data/posttrain/tools.json
git add scripts/tool_eval.py scripts/brain_quiz.py scripts/selfknow_probe.py scripts/toolresult_eval.py \
        scripts/gen_searchqa_data.py scripts/gen_toolresult_data.py scripts/demo/build_search_pack.py scripts/demo/tools_exec.py
git add configs/brain_mech13.yaml configs/sft_v4_reh13_v2.yaml configs/sft_v4_reh13_v2tr.yaml
git add -f temp/n57_* temp/queue_5080_brainmech.sh temp/queue_5080_searchqa.sh temp/queue_5080_toolresult.sh \
           temp/queue_5080_treval.sh temp/queue_5090_sftarm.sh temp/test_tool_proto2.py temp/test_tools_exec.py
git commit -q -F temp/n58_commit_msg.txt
git show --stat --format='%h %s' HEAD | tail -50
echo "== left uncommitted =="
git status --short | grep -v '^?? logs/' | grep -v '^ M logs/'
