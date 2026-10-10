#!/bin/bash
# redo_evals_p26za.sh -- 2026-10-10 (Neuro 54). One-shot: r25 evals of screen_p26za_7L_RC3_x50 again after the eval
# scripts learned data_prelut (temp/patch_eval_prelut.py). Runs beside p26zb on the 5080 (small 7L evals).
# Launch: setsid -f bash temp/redo_evals_p26za.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro; LOG=logs/gap_queue.log; r=screen_p26za_7L_RC3_x50
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
GPU=1 T=2048 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
c1=$(grep "r25-evals $r .*chunk1" $LOG | tail -1 | sed -n 's/.*| chunk1 *\([0-9.]*\).*/\1/p')
hay=$(grep "r25-evals $r .*N8:" $LOG | tail -1 | sed -n 's/.*\(N8:.*\)/\1/p' | cut -c1-90)
if [ -n "$c1" ]; then bash temp/alert.sh INFO "$r evals redone (prelut fix): chunk1 $c1 | $hay"
else bash temp/alert.sh ALERT "$r evals still print no chunk1 after the prelut fix"; fi
