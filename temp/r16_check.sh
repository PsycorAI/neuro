#!/bin/bash
# r16 build step 2: parity (flags off = bit-identical) + functional tests on CPU, plus r15 parity re-check.
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=""
PY=/home/glenn/projects/bdh/venv/bin/python
$PY -m py_compile src/model.py src/hybrid_attn.py src/engram_v2.py src/train_gpu.py && echo "[OK] py_compile"
$PY temp/parity_r16_flags.py check
echo "--- r15 parity (reference saved before the r15 change) ---"
$PY temp/parity_r15_flags.py check 2>&1 | tail -3
