#!/bin/bash
# wait_p26e_end.sh -- 2026-10-06. When p26e training exits: log it, mark DONE if it finished, and leave the 5080 free.
# Evals are deferred until Glenn is done with the 5080 (run temp/eval_p26e_deferred.sh then).
cd /home/glenn/projects/neuro
r=screen_p26e_7L_xscatter
while pgrep -f "train_gpu.py --config configs/$r.yaml" > /dev/null; do sleep 15; done
vp=$(grep 'eval\] val_ppl' logs/$r.log | awk '{print $3}' | tr '\n' ' ')
if grep -q '^done:' logs/$r.log; then touch models/checkpoints/$r/DONE; st=finished; else st="ENDED WITHOUT done: line (check logs/$r.log)"; fi
echo "[s1 $(date '+%m-%d %H:%M:%S')] [$r] training $st :: $vp -- 5080 FREE for Glenn; evals deferred (temp/eval_p26e_deferred.sh)" >> logs/gap_queue.log
