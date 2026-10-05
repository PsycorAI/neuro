#!/bin/bash
# bench_h200.sh -- ~1 h throughput benchmark on the rented H200 (2026-10-04). Same protocol as the home 5090
# timings (temp/profile_train.py, whole-model compile, median step after warm-up). Same tokens/step (32k) in each arm.
# Config: configs/v4_cloud_bench.yaml (temp/mk_bench_h200.py): 1B r19d shape, 5% MLP, x1 gap MQAR on every step.
cd /workspace/neuro
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/workspace/venv/bin/python
b(){ local tag=$1; shift
  $P temp/profile_train.py --config configs/v4_cloud_bench.yaml --tag $tag --noprof --wait 6 --active 10 "$@" \
     > logs/bench_$tag.log 2>&1
  echo "$tag exit=$? $(grep -E '^\[prof' logs/bench_$tag.log | head -1) $(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/bench_$tag.log | tail -1)" | tee -a logs/bench_h200.log
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml; }
C="compile=true compile_dynamic=false"
b b32ga2      --batch 32 --grad_accum 2 --set $C                        # MQAR every micro-batch (worst case)
b b32ga2_once --batch 32 --grad_accum 2 --set $C mqar_once_per_step=true
b b64ga1      --batch 64 --grad_accum 1 --set $C
b b32ga2_nomq --batch 32 --grad_accum 2 --set $C mqar_aug=false          # upper bound, no MQAR aux
b b16ga4_once --batch 16 --grad_accum 4 --set $C mqar_once_per_step=true
echo "tokens/day at the best = 86400 x tok/s; 16B tokens needs >= 18.5k tok/s for 10 days" | tee -a logs/bench_h200.log
