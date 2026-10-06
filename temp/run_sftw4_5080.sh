#!/bin/bash
# run_sftw4_5080.sh -- 2026-09-26 evening: tool_eval w005 on the 5080 once the 5090 finishes
# training it, then exit (Glenn needs the 5080 at 19:00). Skips the eval if w005 lands too late.
export CUDA_VISIBLE_DEVICES=1
source /home/glenn/projects/neuro/temp/sftw_lib.sh
say "5080 runner v4 START (eval w005 only)"
# only lines after the 5090 v4 start count (an old "arm w005 exit=143" is in the log)
v4(){ sed -n '/5090 runner v4 START/,$p' "$SLOG"; }
until v4 | grep -q "arm w005 exit="; do sleep 30; done
# wait for the 5090's post-arm recall_eval to finish reading the ckpt (not strictly needed; cheap)
sleep 90
if [ "$(date +%H%M)" -gt 1820 ]; then say "5080 SKIP w005 eval (too late for 19:00)"; exit 0; fi
trained w005 && eval_arm w005
say "5080 DONE"
