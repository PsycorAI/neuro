#!/bin/bash
# list_engram_cfgs.sh -- 2026-10-01 code review: vocab and Engram v2 table size for flagship/v4 configs
cd /home/glenn/projects/neuro
for f in $(grep -lE 'engram_v2_enabled: true' configs/flagship_v3*.yaml configs/*v4*.yaml 2>/dev/null); do
  echo "$f | $(grep -E '^vocab:' $f) | $(grep -E '^engram_v2_K:' $f) | $(grep -E '^engram_v2_heads:' $f)"
done
