#!/bin/bash
# n57_memtest.sh start|status -- 2026-10-12 (Neuro 57, Glenn OK 10-12). Memory test of the v4 cloud run ON the
# instance, next to the trainer: the r25 evals (chunk perplexity with memory on / off, cross-chunk haystack, nl_fact,
# xcopy) on the newest checkpoint. The checkpoint is hard-linked to /workspace/n57_eval/ for the duration (the trainer
# prunes its own copy after ~66 min) and the link is removed at the end. 45 min limit per eval. Log on the instance:
# logs/n57_memtest.log
cd /home/glenn/projects/neuro || exit 1
. temp/n56_host.env
case "$1" in
  start)  $SSH root@$H 'cd /workspace/neuro && mkdir -p /workspace/n57_eval && ck=$(ls models/checkpoints/v4_cloud/step_*.pt | tail -1) && rm -f /workspace/n57_eval/*.pt && ln $ck /workspace/n57_eval/ && ls -la /workspace/n57_eval && setsid -f bash -c "GPU=0 T=2048 PY=/workspace/venv/bin/python bash temp/run_r25_evals.sh v4=/workspace/n57_eval/$(basename $ck) > logs/n57_memtest.log 2>&1; echo \"[memtest] exit=\$? \$(date -u +%H:%M)\" >> logs/n57_memtest.log; rm -f /workspace/n57_eval/*.pt" < /dev/null > /dev/null 2>&1; echo started' 2>&1 ;;
  status) $SSH root@$H 'cd /workspace/neuro; cat logs/n57_memtest.log | cut -c1-420; pgrep -af "temp/(chunk|xchunk_haystack|nl_fact|xcopy)_eval" | cut -c1-90; nvidia-smi --query-gpu=memory.used --format=csv,noheader; grep -E "^step" logs/v4_cloud.log | tail -n 2 | cut -c1-80; tail -n 2 logs/run_v4.log | cut -c1-100' 2>/dev/null ;;
esac
