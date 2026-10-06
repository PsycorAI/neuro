#!/bin/bash
# toks_compare.sh -- mean tok/s, params and wall time per screen log. Usage: bash temp/toks_compare.sh <run> [<run> ...]
cd /home/glenn/projects/neuro
for r in "$@"; do
  t=$(grep -E '^step ' logs/$r.log | awk '{gsub(",","",$10); s+=$10; n++} END {printf "%.0f", s/n}')
  p=$(grep -m1 -oE 'params=[0-9.]+M' logs/$r.log)
  echo "$r  mean_tok/s=$t  $p"
done
