#!/usr/bin/env bash
# ST long-context go/no-go. Trains stateful + random (block 2048, L4, lam 0.99)
# at one scale, then evals both at context 512 / 1024 / 2048.
#   bash scripts/run_st_longctx.sh 9m    # ~1-1.5h, view before sleep
#   bash scripts/run_st_longctx.sh 22m   # ~3-4h, run overnight
# Always on the 5080 (GPU 1) so the 5090's 350M run is untouched.
set -e
SCALE="${1:?usage: run_st_longctx.sh <9m|22m>}"
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=/home/glenn/projects/neuro/.compile_cache
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
stamp() { date -Iseconds; }

ST="st_lc_${SCALE}_stateful"
RND="st_lc_${SCALE}_random"

echo "[longctx ${SCALE}] started $(stamp)"

echo ""; echo "===== RANDOM control ($(stamp)) ====="
rm -rf "models/checkpoints/${RND}"
"$PY" -u src/train_5080_gpu.py --config "configs/${RND}.yaml" --device cuda \
    2>&1 | tee "logs_${RND}.txt"

echo ""; echo "===== STATEFUL ($(stamp)) ====="
rm -rf "models/checkpoints/${ST}"
"$PY" -u src/train_5080_gpu.py --config "configs/${ST}.yaml" --device cuda \
    2>&1 | tee "logs_${ST}.txt"

echo ""; echo "===== long-context eval ($(stamp)) ====="
for B in 512 1024 2048; do
  echo "--- eval @ context ${B} ---"
  "$PY" -u scripts/stateful_eval.py --runs "${RND},${ST}" \
      --eval_tokens 300000 --block "$B" 2>&1 | tee "logs_${SCALE}_eval_b${B}.txt"
done

echo ""; echo "[longctx ${SCALE}] done $(stamp)"
echo "Compare the stateful-Δ% and absolute ppl across contexts 512/1024/2048."
echo "WIN = stateful beats random AND its advantage GROWS with context length."
