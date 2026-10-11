#!/bin/bash
# run_r25_evals.sh -- 2026-10-04. The three r25 "beyond the window" evals for one or more name=ckpt args.
# Usage: GPU=0 CHUNK=512 bash temp/run_r25_evals.sh name=ckpt.pt [...]   (CHUNK unset = the model's own attn_chunk)
# 2026-10-11 (Neuro 55): the evals of screen_p26ze took 66 min on the 5080 (earlier 7L screens: 1-9 min) and the two
# that cannot fit a 16 GB card at T2048 printed nothing. -> each eval is cut off after EVAL_TMO seconds (default 900),
# and xchunk_haystack / nl_fact are skipped on cards under 20 GB (SMALL_OK=1 forces them).
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONUNBUFFERED=1
PY=${PY:-/home/glenn/projects/bdh/venv/bin/python}      # 10-11: overridable (instance: PY=/workspace/venv/bin/python)
CH=${CHUNK:+--chunk $CHUNK}
TMO="timeout ${EVAL_TMO:-900}"
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i ${GPU:-0} 2>/dev/null | tr -dc '0-9')
small=0; [ -n "$mem" ] && [ "$mem" -lt 20000 ] && [ "${SMALL_OK:-0}" != 1 ] && small=1
$TMO $PY temp/chunk_eval.py $CH --iters ${ITERS:-20} --T ${T:-1024} "$@" 2>&1 | grep -E 'C=|Error|error'
if [ $small = 0 ]; then
  $TMO $PY temp/xchunk_haystack_eval.py $CH "$@" 2>&1 | grep -E 'C=|Error|error'
  $TMO $PY temp/nl_fact_eval.py $CH "$@" 2>&1 | grep -E 'C=|Error|error'
fi
$TMO $PY temp/xcopy_eval.py $CH "$@" 2>&1 | grep -E 'C=|Error|error'
