#!/bin/bash
# kill_at_step.sh -- 2026-10-09 (Neuro 54). Resume-control helper: once the trainer of run $1 has saved its step-$2
# checkpoint, send it ONE SIGTERM (what watchdog_tps.sh does on a spill). The chain (queue_5090_n54.sh) then resumes it.
# Waits up to 3 days for the run to get there. The pgrep pattern starts with the python script path (no self-match).
# Launch: setsid -f bash temp/kill_at_step.sh <run> <step> < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro
r=$1; ck=models/checkpoints/$r/$(printf 'step_%07d.pt' $2); end=$(( $(date +%s) + 259200 ))
until { [ -f $ck ] && grep -q "saved .*$(basename $ck)" logs/$r.log 2>/dev/null; } || [ "$(date +%s)" -ge "$end" ]; do sleep 20; done
[ -f $ck ] || { bash temp/alert.sh WARN "kill_at_step: $r never reached step $2"; exit 0; }
sleep 45                                            # let the async ckpt write finish and a few more steps run
bash temp/alert.sh INFO "$r: PLANNED stop after step $2 (resume control) -- the exit=143 alert that follows is expected"
echo "[n54 $(date '+%m-%d %H:%M:%S')] [$r] planned stop after step $2 (resume control)" >> logs/gap_queue.log
pkill -TERM -f "src/train_gpu.py --config configs/$r.yaml"
