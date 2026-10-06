#!/bin/bash
# run_r25_evals.sh -- 2026-10-04. The three r25 "beyond the window" evals for one or more name=ckpt args.
# Usage: GPU=0 CHUNK=512 bash temp/run_r25_evals.sh name=ckpt.pt [...]   (CHUNK unset = the model's own attn_chunk)
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
CH=${CHUNK:+--chunk $CHUNK}
$PY temp/chunk_eval.py $CH --iters ${ITERS:-20} --T ${T:-1024} "$@" 2>&1 | grep -E 'C=|Error|error'
$PY temp/xchunk_haystack_eval.py $CH "$@" 2>&1 | grep -E 'C=|Error|error'
$PY temp/nl_fact_eval.py $CH "$@" 2>&1 | grep -E 'C=|Error|error'
$PY temp/xcopy_eval.py $CH "$@" 2>&1 | grep -E 'C=|Error|error'
