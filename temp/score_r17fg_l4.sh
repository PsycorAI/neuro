#!/bin/bash
# score_r17fg_l4.sh -- 2026-10-02: KG recall + interpretability scorecard (CPU) for r17f, r17g, r16 l4.
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
CK=models/checkpoints
A="r17f=$CK/screen_r17f_k688_s0_30m/step_0006104.pt r17g=$CK/screen_r17g_ramp03_s0_30m/step_0006104.pt l4=$CK/screen_r16_l4_lev4_s0_30m/step_0006104.pt"
$PY scripts/kg_recall_eval.py --device cpu --B 32 --reps 2 $A > logs/kg_r17fg_l4.log 2>&1
$PY scripts/interp_scorecard.py --device cpu --B 4 $A > logs/scorecard_r17fg_l4.log 2>&1
echo SCORE_DONE >> logs/scorecard_r17fg_l4.log
