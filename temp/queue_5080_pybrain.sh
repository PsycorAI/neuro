#!/bin/bash
# queue_5080_pybrain.sh [gpu=1] -- 2026-10-12 (Neuro 58; Glenn: Python brain = brain OFF cannot write the script,
# brain ON can). First test at 13L: a LoRA adapter (r 32 on qkv / o_proj / ffn) trained on the synthetic Python course
# (scripts/gen_pybrain_data.py: 2,700 study lines, every solution executed) on top of sft_v4_reh13_v2tr, then the
# run-the-code quiz (scripts/brain_code_quiz.py): base vs brain, taught families in unseen wordings vs control families.
# Results: logs/brain_py13.log, logs/brain_py13_quiz.log, one DONE / ALERT.
# Launch: setsid -f bash temp/queue_5080_pybrain.sh 1 < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
GPU=${1:-1}; R=brain_py13
PY=/home/glenn/projects/bdh/venv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
A(){ bash temp/alert.sh "$@"; }
$PY scripts/gen_pybrain_data.py > logs/${R}_data.log 2>&1 || { A ALERT "$R: data generation failed (logs/${R}_data.log)"; exit 1; }
$PY temp/n57_mk_arm.py $R sft_init=sft_v4_reh13_v2tr tool_proto=2 lora_enabled=true lora_r=32 lora_alpha=64 "lora_targets=[qkv, o_proj, ffn]" \
  "sft_jsonl=[/home/glenn/projects/neuro/training_data/brains/python/study.jsonl]" sft_dataset=null sft_max_len=256 \
  batch_size=8 grad_accum=4 lr=0.0002 max_tokens=4000000 ckpt_every=200 gen_check_every=200 log_every=20 >> logs/${R}_data.log 2>&1
touch logs/$R.log
A INFO "$R (13L Python brain test) started on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU timeout 7200 $PY src/train_sft.py --config configs/$R.yaml --device cuda >> logs/$R.log 2>&1
grep -q '^done:' logs/$R.log || { A ALERT "$R did NOT finish: $(grep -E 'Error|Traceback' logs/$R.log | tail -1 | cut -c1-160)"; exit 1; }
CUDA_VISIBLE_DEVICES=$GPU timeout 3600 $PY scripts/brain_code_quiz.py --base sft_v4_reh13_v2tr --quiz training_data/brains/python/quiz.jsonl \
  --brain py=$R --n 90 > logs/${R}_quiz.log 2>&1
q=$(grep '^  quiz' logs/${R}_quiz.log | sed 's/^  quiz //' | tr '\n' ';' | cut -c1-230)
if [ -z "$q" ]; then A ALERT "$R trained but the quiz gave no result (logs/${R}_quiz.log)"
else A DONE "$R Python brain (code that passes its tests): $q"; fi
