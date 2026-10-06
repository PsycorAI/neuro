#!/bin/bash
# qv_sens.sh -- 2026-10-06: zero-shot cost of spike-coding attention q/v on the converted r25t (all thetas recalibrated
# in both arms, so the comparison is like for like).
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=0
PY=/home/glenn/projects/bdh/venv/bin/python
SRC=$(ls models/checkpoints/r25t_13L_spk_L8_80M_150m/step_*.pt | tail -1)
cp configs/r25t_13L_spk_L8_80M_150m.yaml /tmp/qv_a.yaml
sed 's/^- mem$/- mem\n- q\n- v/' configs/r25t_13L_spk_L8_80M_150m.yaml > /tmp/qv_b.yaml
grep -A8 '^spike_code:' /tmp/qv_b.yaml | tr '\n' ' '; echo
for cfg in /tmp/qv_a.yaml /tmp/qv_b.yaml; do
  echo -n "$cfg: "; $PY temp/mk_spike_seed.py $SRC /tmp/qv_seed $cfg 1.0 2>&1 | grep -oE 'dense [0-9.]+ -> coded [0-9.]+ \| mean spikes/input [0-9.]+'
done
