#!/bin/bash
# cmp_2L_3L.sh -- 2026-09-30 read-only: compare 2L vs 3L signed training logs (params, tok/s, loss, grad, spike stats)
cd /home/glenn/projects/neuro
for r in screen_r8_signed_s1_30m screen_r10_signed_3L_s1_30m screen_r5_full_s1_30m screen_r5_3L_s0_30m screen_tfmr3_adamw_s0_30m; do
  echo "=== $r"
  grep -m3 -iE 'param' logs/$r.log | cut -c1-160
  grep -E '^step (100|500|1000|2000|3000|6000) ' logs/$r.log | cut -c1-160
  grep -E 'val_ppl' logs/$r.log | tail -1 | cut -c1-160
  grep -iE '\[spike\]|\[grad|gnorm|grad_norm' logs/$r.log | tail -2 | cut -c1-300
done
