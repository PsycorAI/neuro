#!/bin/bash
# spk_site_sens.sh -- 2026-10-06: zero-shot ppl when spike-coding ONE site at a time (p25o ckpt), at 2 calibration scales.
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=0
PY=/home/glenn/projects/bdh/venv/bin/python
SRC=models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt
for site in down; do
  for lv in 4 8; do
    sed "s/^spike_code:.*//" configs/screen_p25s_7L_spk_conv.yaml | grep -v '^- \(qkv\|o\|up\|down\|head\|mem\)$' > /tmp/sens.yaml
    printf "spike_code:\n- $site\nspike_levels: $lv\nspike_levels_hidden: $((lv * 2))\n" >> /tmp/sens.yaml
    sed -i '0,/^spike_levels: 4$/{/^spike_levels: 4$/d}; 0,/^spike_levels_hidden: 8$/{/^spike_levels_hidden: 8$/d}' /tmp/sens.yaml
    echo -n "$site L$lv: "; $PY temp/mk_spike_seed.py $SRC /tmp/sens_seed /tmp/sens.yaml 1.0 2>&1 | grep -oE 'dense [0-9.]+ -> coded [0-9.]+ \| mean spikes/input [0-9.]+'
  done
done
