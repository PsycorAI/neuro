#!/bin/bash
# queue_5080_fdcbrain.sh [gpu=1] -- 2026-10-12 (Neuro 58; Glenn: health brain = USDA FoodData Central). First test at
# 13L: a LoRA adapter (r 32 on qkv / o_proj / ffn) trained on calories / protein / carbohydrate per 100 g of 150 raw
# foods (scripts/gen_fdcbrain_data.py: 450 facts, 8 wordings each) on top of sft_v4_reh13_v2tr, then the closed-book
# quiz (scripts/brain_quiz.py): base vs brain, taught foods in unseen wordings vs 30 control foods never taught.
# Results: logs/brain_fdc13.log, logs/brain_fdc13_quiz.log, one DONE / ALERT.
# Launch: setsid -f bash temp/queue_5080_fdcbrain.sh 1 < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
GPU=${1:-1}; R=brain_fdc13
PY=/home/glenn/projects/bdh/venv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
A(){ bash temp/alert.sh "$@"; }
/usr/bin/python3 -I scripts/gen_fdcbrain_data.py > logs/${R}_data.log 2>&1 || { A ALERT "$R: data generation failed (logs/${R}_data.log)"; exit 1; }
$PY temp/n57_mk_arm.py $R sft_init=sft_v4_reh13_v2tr tool_proto=2 lora_enabled=true lora_r=32 lora_alpha=64 "lora_targets=[qkv, o_proj, ffn]" \
  "sft_jsonl=[/home/glenn/projects/neuro/training_data/brains/health/study.jsonl]" sft_dataset=null sft_max_len=256 \
  batch_size=8 grad_accum=4 lr=0.0002 max_tokens=8000000 ckpt_every=200 gen_check_every=200 log_every=20 >> logs/${R}_data.log 2>&1
touch logs/$R.log
A INFO "$R (13L health brain test) started on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU timeout 10800 $PY src/train_sft.py --config configs/$R.yaml --device cuda >> logs/$R.log 2>&1
grep -q '^done:' logs/$R.log || { A ALERT "$R did NOT finish: $(grep -E 'Error|Traceback' logs/$R.log | tail -1 | cut -c1-160)"; exit 1; }
CUDA_VISIBLE_DEVICES=$GPU timeout 3600 $PY scripts/brain_quiz.py --base sft_v4_reh13_v2tr --quiz training_data/brains/health/quiz.jsonl \
  --brain health=$R --n 150 > logs/${R}_quiz.log 2>&1
q=$(grep '^  quiz' logs/${R}_quiz.log | sed 's/^  quiz //' | tr '\n' ';' | cut -c1-230)
if [ -z "$q" ]; then A ALERT "$R trained but the quiz gave no result (logs/${R}_quiz.log)"
else A DONE "$R health brain (exact USDA values): $q"; fi
