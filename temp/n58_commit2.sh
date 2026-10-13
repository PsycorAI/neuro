#!/bin/bash
# Neuro 58: commit today's work (no logs, no data/, no checkpoints, no push).
cd /home/glenn/projects/neuro || exit 1
set -e
git add docs/to-do.html src/muon.py src/train_gpu.py
git add scripts/gen_pybrain_data.py scripts/gen_fdcbrain_data.py scripts/brain_code_quiz.py scripts/merge_brain.py
git add configs/probe_v4_*.yaml configs/brain_py13.yaml configs/brain_fdc13.yaml configs/sft_v4_dense_speed.yaml
git add -f temp/n58_* temp/queue_5090_n58*.sh temp/queue_5080_pybrain.sh temp/queue_5080_fdcbrain.sh temp/queue_5080_brainon.sh \
           temp/test_no_decay_remap.py temp/xchunk_haystack_eval.py
git commit -q -F temp/n58_commit_msg2.txt
git show --stat --format='%h %s' HEAD | tail -n 6
echo "== left uncommitted =="
git status --short | grep -v -E '^\?\? logs/|^ M logs/' | head -n 12
