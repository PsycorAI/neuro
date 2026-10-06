#!/bin/bash
# run_sftw2_5080.sh -- 2026-09-26: replaces run_sftw_5080.sh mid-run (killed while its B8 probe
# ran) to add a w0.05 arm: recall held at w0.1 (62/25), so find the knee below it.
# 1) wait for the orphaned probe_tfmr426_b8 python, log it; 2) B4 probe; 3) eval w01;
# 4) train w005 on the 5080 + recall; 5) drain the shared tool-eval queue (+ w005).
export CUDA_VISIBLE_DEVICES=1
source /home/glenn/projects/neuro/temp/sftw_lib.sh
EVAL_ARMS="$EVAL_ARMS w005"
say "5080 runner v2 START (adds w005)"
while pgrep -f 'train_gpu.py --config temp/[p]robe_tfmr426_b8' > /dev/null; do sleep 20; done
pkill -f 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i [1]'
run=probe_tfmr426_b8
say "$run done peak_sys_MiB=$(sort -n logs/$run.smi | tail -n 1) :: $(grep -E '^step' logs/$run.log | tail -n 1) :: $(grep -cE 'out of memory' logs/$run.log) OOM lines"
rm -rf models/checkpoints/$run
run=probe_tfmr426_b4
rm -rf models/checkpoints/$run
say "$run START"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 -lms 2000 > logs/$run.smi &
smi=$!
$PY src/train_gpu.py --config temp/$run.yaml --device cuda > logs/$run.log 2>&1
rc=$?
kill $smi
say "$run exit=$rc peak_sys_MiB=$(sort -n logs/$run.smi | tail -n 1) :: $(grep -E '^step' logs/$run.log | tail -n 1)"
rm -rf models/checkpoints/$run
trained w01 && eval_arm w01
say "arm w005 START"
$PY src/train_sft.py --config configs/screen_sftab_w005_363m.yaml --device cuda > logs/screen_sftab_w005.log 2>&1
say "arm w005 exit=$?"
$PY scripts/recall_eval.py --runs flagship_v3_363m,screen_sftab_w005_363m --B 32 2>&1 | grep -E '^ *(run|flagship|screen|chance)' >> "$SLOG"
eval_until_done
say "5080 DONE"
