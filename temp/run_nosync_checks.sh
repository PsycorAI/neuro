#!/bin/bash
# run_nosync_checks.sh -- 2026-09-27: (1) 30M eager determinism check baseline vs --lif_nosync
# (losses must match exactly); (2) repeat v4-draft B8 baseline timing (noise estimate).
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
P="temp/profile_train.py --config configs/screen_indepwin_s0_30m.yaml --batch 32 --wait 20 --active 100 --noprof"
$PY $P --tag 30m_base > logs/prof_30m_base.log 2>&1
$PY $P --tag 30m_nosync --lif_nosync > logs/prof_30m_nosync.log 2>&1
rm -rf models/checkpoints/screen_prof_30m_base models/checkpoints/screen_prof_30m_nosync
grep -E '^step' logs/prof_30m_base.log | cut -d'|' -f1,3 > /tmp/a30.txt
grep -E '^step' logs/prof_30m_nosync.log | cut -d'|' -f1,3 > /tmp/b30.txt
if cmp -s /tmp/a30.txt /tmp/b30.txt; then echo "30M LOSSES IDENTICAL ($(wc -l < /tmp/a30.txt) log lines)"; else echo "30M LOSSES DIFFER"; diff /tmp/a30.txt /tmp/b30.txt | head; fi
grep -h 'prof 30m' logs/prof_30m_base.log logs/prof_30m_nosync.log
$PY temp/profile_train.py --config configs/flagship_v4_363m.draft.yaml --tag v4d_b8_np2 --batch 8 --wait 40 --active 20 --noprof > logs/prof_v4d_b8_np2.log 2>&1
rm -rf models/checkpoints/screen_prof_v4d_b8_np2
grep -h 'median' logs/prof_v4d_b8_np.log logs/prof_v4d_b8_np2.log logs/prof_v4d_b8_nosync.log
