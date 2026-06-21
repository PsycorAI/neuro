#!/usr/bin/env bash
# Today's 5080 plan (4 experiments, ~3-4 hours total).
# Replaces the abandoned overnight_5080.sh (whose 100M-brain step was infeasible).
# Launch with:
#   nohup bash scripts/today_5080.sh > today_log.txt 2>&1 &
#   tail -f today_log.txt
set -e
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python

export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=/home/glenn/projects/neuro/.compile_cache
export PYTHONUNBUFFERED=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
export CUDA_VISIBLE_DEVICES=1

stamp() { date -Iseconds; }

echo "============================================================"
echo "[today_5080] started at $(stamp)"
echo "============================================================"

# ---- 1. Domain-transfer brain test (highest strategic value) ------------
echo ""
echo "=========================================================="
echo "[1/4] Domain-transfer brain test (200K tok per brain)"
echo "  starts: $(stamp)"
echo "=========================================================="
"$PY" -u scripts/brain_domain_transfer.py --exposure_tokens 200000 \
    2>&1 | tee logs_brain_domain_transfer.txt
echo "[1/4] done: $(stamp)"

# ---- 2. Lighter KD A/B (alpha=0.3, T=1.0) -------------------------------
echo ""
echo "=========================================================="
echo "[2/4] KD light config (alpha=0.3, T=1.0)"
echo "  starts: $(stamp)"
echo "=========================================================="
"$PY" -u src/train_5080_gpu.py --config configs/phase25_a_kd_light.yaml \
    --device cuda 2>&1 | tee logs_phase25_a_kd_light.txt
echo "[2/4] done: $(stamp)"

# ---- 3. Heavy-sparsity 9M run -------------------------------------------
echo ""
echo "=========================================================="
echo "[3/4] heavy-sparsity 9M run (sparsity_lambda=2.0)"
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
echo "[3/4] done: $(stamp)"

# ---- 4. Extended brain test at native λ=0.98 (1M tokens) ----------------
echo ""
echo "=========================================================="
echo "[4/4] λ=0.98 brain test, longer exposure (1M tokens)"
echo "  starts: $(stamp)"
echo "=========================================================="
"$PY" -u scripts/brain_slow_decay_test.py --exposure_tokens 1000000 \
    2>&1 | tee logs_brain_slow_decay_long.txt
echo "[4/4] done: $(stamp)"

echo ""
echo "============================================================"
echo "[today_5080] all four experiments finished at $(stamp)"
echo "============================================================"
echo ""
echo "Result files:"
echo "  logs_brain_domain_transfer.txt     (KEY: brain domain specificity?)"
echo "  logs_phase25_a_kd_light.txt        (does gentler KD beat noKD?)"
echo "  logs_phase25_a_heavy_sparsity.txt  (does heavy penalty fix spike rate?)"
echo "  logs_brain_slow_decay_long.txt     (does λ=0.98 brain plateau at 1M?)"
