#!/bin/bash
cd /home/glenn/projects/neuro/logs
for f in indepwin_screen.log screen_lifx_g1_t18.log screen_lifx_g0_t18.log screen_lifx_fptK3.log screen_lifx_fptK1.log; do
  printf '%-26s' "$f"; grep -E '^step [0-9]+ ' "$f" | head -10 | awk '{printf "%s:%s ", $2, $8}'; echo
done
