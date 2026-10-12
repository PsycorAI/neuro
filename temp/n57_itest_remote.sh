#!/bin/bash
# n57_itest_remote.sh <ckpt> -- 2026-10-12 (Neuro 57, Glenn OK). Runs ON the instance next to the trainer (~27 GB free):
# small-batch learning checks of one v4 checkpoint. Each step under a 30 min limit. Log: logs/n57_itest.log
cd /workspace/neuro || exit 1
CK=${1:?ckpt}; PY=/workspace/venv/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HAYSTACK_B=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run(){ echo "== $1"; shift; timeout 1800 "$@" 2>&1 | grep -v -i warning | grep -E "SAMPLE|v4|C=|energy|recall|Error|error|%|N8|N32" | cut -c1-600; }
{
run "text samples" $PY temp/n57_sample.py $CK
run "haystack in-window (gap 0/64/192/440)" $PY temp/haystack_recall_eval.py v4=$CK
run "nl_fact (n 32)" $PY temp/nl_fact_eval.py --n 32 v4=$CK
run "cross-chunk haystack" $PY temp/xchunk_haystack_eval.py v4=$CK
run "energy-v2" $PY temp/energy_eval_v2.py v4=$CK
echo "[itest] done $(date -u +%H:%M)"
} > logs/n57_itest.log 2>&1
