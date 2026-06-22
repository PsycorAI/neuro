#!/usr/bin/env bash
# KD-refine the flagship 350M on the 5090 (GPU 0). Resumes the model-only seed
# in models/checkpoints/phase3_350M_kd (see temp/seed_kd_ckpt.py) and runs a
# gentle Llama-3.2-1B distillation pass to +0.5B tokens.
#   bash scripts/run_kd_350m.sh > logs_phase3_350M_kd.txt 2>&1 &
#   tail -f logs_phase3_350M_kd.txt
set -e
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
echo "===== run_kd_350m start $(date -Iseconds) ====="
"$PY" -u src/train_5090_gpu.py --config configs/phase3_350M_kd.yaml --device cuda
echo "===== run_kd_350m done $(date -Iseconds) ====="
