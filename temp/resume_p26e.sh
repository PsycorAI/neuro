#!/bin/bash
# resume_p26e.sh -- 2026-10-06 (Glenn: 5080 free now, no idle wait). Resume p26e (auto-resume from step 5299), then evals.
cd /home/glenn/projects/neuro
GPU=1 FREE_MIN=0 UNTIL="2026-10-07 06:00" bash temp/run_5080_sched.sh screen_p26e_7L_xscatter   # + recall/haystack
[ -f models/checkpoints/screen_p26e_7L_xscatter/DONE ] && GPU=1 SKIP_RH=1 bash temp/eval_p26e_deferred.sh
