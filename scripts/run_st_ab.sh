#!/usr/bin/env bash
# ST Phase 1.3 — stateful vs random-window A/B at 9M scale.
# Runs both arms back-to-back on the 5080, same 100M-token budget.
#   nohup bash scripts/run_st_ab.sh > logs_st_ab.txt 2>&1 &
#   tail -f logs_st_ab.txt
set -e
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=/home/glenn/projects/neuro/.compile_cache
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
stamp() { date -Iseconds; }

echo "[ST A/B] started $(stamp)"

echo ""
echo "===== ARM 1: random-window control ($(stamp)) ====="
# Clean any prior checkpoints so this arm starts FRESH (auto-resume would
# otherwise contaminate the A/B by continuing old weights).
rm -rf models/checkpoints/phase25_st_random
"$PY" -u src/train_5080_gpu.py --config configs/phase25_st_random.yaml \
    --device cuda 2>&1 | tee logs_st_random.txt

echo ""
echo "===== ARM 2: stateful streaming ($(stamp)) ====="
rm -rf models/checkpoints/phase25_st_stateful
"$PY" -u src/train_5080_gpu.py --config configs/phase25_st_stateful.yaml \
    --device cuda 2>&1 | tee logs_st_stateful.txt

echo ""
echo "[ST A/B] both arms done $(stamp)"
echo "Compare final val_ppl:"
echo "  random:   $(grep -oE 'val_ppl [0-9.]+' logs_st_random.txt | tail -1)"
echo "  stateful: $(grep -oE 'val_ppl [0-9.]+' logs_st_stateful.txt | tail -1)"
