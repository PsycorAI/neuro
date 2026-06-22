#!/usr/bin/env bash
# ST larger-budget test (#2) + Phase 6 resets (#1), 300M tokens, 9M/4-layer.
# Isolates: does more data (gated vs random) and/or anti-bleeding resets
# (gated+reset) let stateful finally beat the random baseline?
#   bash scripts/run_st_big.sh > logs_st_big.txt 2>&1 &
#   tail -f logs_st_big.txt
set -e
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=/home/glenn/projects/neuro/.compile_cache
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
stamp() { date -Iseconds; }

for cfg in st_big_random st_big_gated st_big_gated_reset; do
  echo ""; echo "===== ${cfg} ($(stamp)) ====="
  rm -rf "models/checkpoints/${cfg}"
  "$PY" -u src/train_5080_gpu.py --config "configs/${cfg}.yaml" --device cuda \
      2>&1 | tee "logs_${cfg}.txt"
done

echo ""; echo "===== 3-way eval ($(stamp)) ====="
for B in 512 2048; do
  echo "--- ctx ${B} ---"
  "$PY" -u scripts/stateful_eval.py \
      --runs st_big_random,st_big_gated,st_big_gated_reset \
      --eval_tokens 300000 --block "$B" 2>&1 | grep -vE "^Reading|slope|^saved curve"
done

echo ""; echo "===== learned alpha spread (gated arms) ($(stamp)) ====="
"$PY" -u -c "
import torch, glob
for run in ['st_big_gated','st_big_gated_reset']:
    p=sorted(glob.glob(f'models/checkpoints/{run}/step_*.pt'))[-1]
    b=torch.load(p, map_location='cpu', weights_only=False)
    for k,v in b['model'].items():
        if 'blocks.0.decay_raw' in k:
            a=torch.sigmoid(v); print(f'{run}: alpha {a.min():.3f}..{a.max():.3f} mean {a.mean():.3f}')
"
echo "[done $(stamp)]"
