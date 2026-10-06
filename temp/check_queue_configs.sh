#!/bin/bash
# 2026-09-29: verify every config in the signed-first queue exists
cd /home/glenn/projects/neuro
for r in screen_r7_gap128_s0_30m screen_r8_signed_s1_30m screen_r8_signed_lam2_s0_30m screen_r9a_stateful_s0_30m \
         screen_r9b_swastoch_s0_30m screen_r9c_stateful_swa_s0_30m screen_r8_signed_3L_s0_30m \
         screen_r8_signed_canon_s0_30m screen_r7_gapw015_s0_30m; do
  if [ -f configs/$r.yaml ]; then echo "ok      $r"; else echo "MISSING $r"; fi
done
