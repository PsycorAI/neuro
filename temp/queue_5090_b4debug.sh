#!/bin/bash
# queue_5090_b4debug.sh -- 2026-10-02 (Glenn OK): bisect the Inductor codegen crash
# ("get_block_shape ... list indices must be integers or slices, not NoneType") seen compiling probe_g4_c0s1 at B4
# (B8 compiles fine). Compile-only (pass/fail), not timing. Each variant shrinks one 1B setting.
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/b4debug_5090.log
say(){ echo "[b4 $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
exec 8>logs/5090_seq.lock
flock 8
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
try(){   # try <tag> <batch> [set k=v ...]
  local tag=$1 b=$2; shift 2
  rm -rf models/checkpoints/screen_prof_$tag
  $PY temp/profile_train.py --config configs/probe_g4_c0s1.yaml --tag $tag --batch $b --noprof --wait 2 --active 2 \
      --set compile=true compile_dynamic=false "$@" > logs/prof_$tag.log 2>&1
  local rc=$?
  say "$tag (B$b $*): exit=$rc $(grep -c 'get_block_shape' logs/prof_$tag.log) crash-lines"
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml
}
try b4_base 4
try b4_b2 2
try b4_b6 6
try b4_engK8k 4 engram_v2_K=8192
try b4_vocab16k 4 vocab=16384
try b4_n2048 4 n_neurons=2048
try b4_ffn2752 4 attn_ffn_hidden=2752 attn_ffn_spikeglu_k=256
try b4_L4 4 n_layers=4 "attention_layer_positions=[1,2,3]"
TORCHDYNAMO_VERBOSE=1 TORCH_LOGS=output_code try b4_verbose 4
say "B4 DEBUG DONE"
