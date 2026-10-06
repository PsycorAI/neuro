#!/bin/bash
# run_5090_seq.sh -- 2026-09-28 5090 sequence (Glenn's priorities): current ppl-gap screens -> 426M transformer short LR
# check (3 arms x 60M tok, ~3h) -> v3 heal + post-training chain (temp/run_5090_v3_chain.sh). One sequence = no GPU races.
# WATCH: tail -f /home/glenn/projects/neuro/logs/gap_5090.log /home/glenn/projects/neuro/logs/v3_chain.log
cd /home/glenn/projects/neuro
exec 8>logs/5090_seq.lock; flock -n 8 || { echo "already running"; exit 1; }
echo "[seq $(date '+%m-%d %H:%M:%S')] waiting for ppl-gap screens" >> logs/gap_5090.log
while pgrep -f "[r]un_5090_gap_stream" > /dev/null; do sleep 60; done
bash temp/run_5090_gap_stream.sh lr screen_tfmr426_lr3e4 screen_tfmr426_lr6e4 screen_tfmr426_lr12e4
bash temp/run_5090_v3_chain.sh
