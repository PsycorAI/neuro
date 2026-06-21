#!/usr/bin/env bash
# Overnight pipeline for the 5080.
# Runs three experiments sequentially. Each section logs to its own file.
# Launch with:
#   nohup bash scripts/overnight_5080.sh > overnight_log.txt 2>&1 &
#   tail -f overnight_log.txt   # to spot-check progress before you sleep
set -e
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python

export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=/home/glenn/projects/neuro/.compile_cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
export CUDA_VISIBLE_DEVICES=1

stamp() { date -Iseconds; }

echo "============================================================"
echo "[overnight] started at $(stamp)"
echo "============================================================"

# ---- 1. Phase 1a slow-decay brain-mod test (highest strategic value) ----
echo ""
echo "=========================================================="
echo "[1/3] Phase 1a slow-decay brain test (100M tok per λ)"
echo "  starts: $(stamp)"
echo "=========================================================="
"$PY" scripts/brain_slow_decay_test.py --exposure_tokens 100000000 \
    2>&1 | tee logs_brain_slow_decay.txt
echo "[1/3] done: $(stamp)"

# ---- 2. Lighter KD A/B (alpha=0.3, T=1.0) ----
echo ""
echo "=========================================================="
echo "[2/3] KD light config (alpha=0.3, T=1.0)"
echo "  starts: $(stamp)"
echo "=========================================================="
"$PY" -u src/train_5080_gpu.py --config configs/phase25_a_kd_light.yaml \
    --device cuda 2>&1 | tee logs_phase25_a_kd_light.txt
echo "[2/3] done: $(stamp)"

# ---- 3. Heavier-sparsity 9M run (sparsity_lambda 2.0, no KD) ----
# Build a one-off config inline so we don't pollute configs/ with a 30 min run.
echo ""
echo "=========================================================="
echo "[3/3] heavy-sparsity 9M run (sparsity_lambda=2.0)"
echo "  starts: $(stamp)"
echo "=========================================================="
HEAVY=configs/phase25_a_heavy_sparsity.yaml
cat > "$HEAVY" <<'YAML'
run_name: phase25_a_heavy_sparsity
arch: spiking
vocab: 16384
d: 256
n_neurons: 512
d_mem: 256
n_layers: 1
recurrent: true
rec_density: 0.02
set_zeta: 0.3
set_every: 200
block_size: 128
batch_size: 128
grad_accum: 1
eval_batch: 64
use_fpt: true
fpt_K: 10
sparsity_lambda: 2.0
sparsity_target: 0.04
optimizer: muon
muon_lr: 0.005
lr: 0.0015
max_tokens: 100000000
eval_every: 2000
ckpt_every: 2000
keep_last: 3
log_every: 1000
amp: true
compile: true
compile_mode: default
compile_dynamic: true
seed: 0
YAML
"$PY" -u src/train_5080_gpu.py --config "$HEAVY" --device cuda \
    2>&1 | tee logs_phase25_a_heavy_sparsity.txt
echo "[3/3] done: $(stamp)"

echo ""
echo "============================================================"
echo "[overnight] all three experiments finished at $(stamp)"
echo "============================================================"
echo ""
echo "Result files to review when you wake up:"
echo "  logs_brain_slow_decay.txt          (Phase 1a — KEY result)"
echo "  logs_phase25_a_kd_light.txt        (KD with gentler hyperparams)"
echo "  logs_phase25_a_heavy_sparsity.txt  (spike-rate fix test)"
echo "  models/brains/slow_decay_test/     (saved brains by λ)"
echo "  models/checkpoints/phase25_a_kd_light/"
echo "  models/checkpoints/phase25_a_heavy_sparsity/"
