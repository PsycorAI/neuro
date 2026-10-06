#!/bin/bash
# run_5080_chain_0926.sh -- 2026-09-26: strictly-serial 5080 queue (sharing the 5080 OOM'd
# depth s1 at step ~3900 tonight). 1) wait for the SFT tool_eval runner, 2) resume the depth
# screen (s1 from step 3000, then its recall eval), 3) the 30M fixed-transformer preview.
cd /home/glenn/projects/neuro
say(){ echo "[chain5080 $(date +%H:%M:%S)] $*" | tee -a logs/chain_5080.log; }
say "waiting for tool_eval runner"
while pgrep -f '[r]un_sftab_tooleval.sh' > /dev/null; do sleep 30; done
say "depth resume"
bash temp/run_5080_depth.sh; say "depth exit=$?"
say "tfmrfix"
bash temp/run_5080_tfmrfix.sh; say "tfmrfix exit=$?"
say "DONE"
