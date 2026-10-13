#!/bin/bash
# n58_ckpt_health.sh <ckpt> [gpu=1] -- 2026-10-12 (Neuro 58). Health check of one pulled v4 checkpoint, on the 5080 by
# default, ONE python at a time: cfg patch, memory-layer spike diagnostics, per-layer FFN / embedding diagnostic,
# dead-unit count, per-chunk perplexity (fixed seed), cross-chunk recall. Before each step it waits (max 3 h) until
# no other checkpoint-loading job of this kind runs and >= 13 GB RAM is available (a 1B checkpoint is 10 GB in RAM;
# see memory reference_wsl_oom_kill_loop). Appends to logs/n58_ckpt_health.log; one DONE alert with the headline.
cd /home/glenn/projects/neuro || exit 1
CK=${1:?ckpt}; N=$(basename $CK .pt); LOG=logs/n58_ckpt_health.log
PY=/home/glenn/projects/bdh/venv/bin/python
export CUDA_VISIBLE_DEVICES=${2:-1} PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HAYSTACK_B=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9> logs/n58_ckpt_health.lock; flock 9                      # one health check at a time
gate(){ end=$(( $(date +%s) + 10800 ))
  until [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 13 ] \
        && ! pgrep -f 'temp/(chunk_eval|xchunk_haystack_eval|nl_fact_eval|n58_revive|n58_spike_diag|n58_layer_diag|n58_patch_cfg)\.py' > /dev/null; do
    [ "$(date +%s)" -ge "$end" ] && return 1; sleep 30; done; }
run(){ gate || { echo "  (skipped: no free RAM for 3 h) $1" >> $LOG; return; }; shift
  timeout 2400 "$@" 2>&1 9>&- | grep -v -i warning | grep -E 'C=|rate|per |code|by chunk|blk|embed|patched|already|rror|Killed' | cut -c1-330 >> $LOG; }
echo "== $CK ($(date '+%m-%d %H:%M'))" >> $LOG
run patch  $PY temp/n58_patch_cfg.py $CK
run spike  $PY temp/n58_spike_diag.py --B 2 $N=$CK
run layer  $PY temp/n58_layer_diag.py --B 2 $N=$CK
run dead   $PY temp/n58_revive.py $CK /tmp/none.pt --dry
run chunk  $PY temp/chunk_eval.py --T 2048 --iters 20 --B 2 $N=$CK
run xchunk $PY temp/xchunk_haystack_eval.py $N=$CK
echo "== done $N ($(date '+%m-%d %H:%M'))" >> $LOG
bash temp/alert.sh DONE "health $N: $(grep -E "^$N .*chunk1" $LOG | tail -n 1 | cut -c28-170) | $(sed -n "/== $(echo $CK | sed 's/\//\\\//g')/,\$p" $LOG | grep -E 'per token' | tail -n 1 | cut -c1-90)" 9>&-
