#!/bin/bash
# Median logged tok/s per r16 run (steps 3000-5900), plus wall time from the queue log.
cd /home/glenn/projects/neuro
for r in "$@"; do
  med=$(grep -E '^step [0-9]+ \|' logs/$r.log | awk -F'|' '{n=split($1,a," "); s=a[2]; if (s>=3000 && s<=5900) {gsub(/[ ,]|tok\/s/,"",$4); print $4}}' | sort -n | awk '{v[NR]=$1} END {print v[int((NR+1)/2)]}')
  echo "$r median_tok_s=$med"
done
