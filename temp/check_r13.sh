#!/bin/bash
# check_r13.sh -- 2026-09-30: generate r13 configs and diff each against its parent
cd /home/glenn/projects/neuro
grep -n depth_scaled configs/screen_r8_signed_3L_s0_30m.yaml configs/screen_r5_signed_s0_30m.yaml
/usr/bin/python3 temp/mk_gap_r13_configs.py || exit 1
echo "== a"; diff configs/screen_r8_signed_3L_s0_30m.yaml configs/screen_r13a_signed_3L_attn2_s0_30m.yaml
echo "== b"; diff configs/screen_r8_signed_3L_s0_30m.yaml configs/screen_r13b_signed_3L_dsinit_s0_30m.yaml
echo "== d"; diff configs/screen_r5_signed_s0_30m.yaml configs/screen_r13d_signed_n4096_s0_30m.yaml
