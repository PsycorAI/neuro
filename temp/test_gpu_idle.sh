#!/bin/bash
# test_gpu_idle.sh -- 2026-10-11 (Neuro 55): the per-GPU idle block of temp/alert_watch.sh with a fake nvidia-smi.
# GPU 0 is idle on every check, GPU 1 is busy on checks 5 and 12 -> expect one WARN for GPU 0 (at check 8) and one for
# GPU 1 only at check 20 (8 idle checks after its last busy one).
A(){ echo "check $n: $*"; }
nvidia-smi(){ echo "0, 225"; if [ $n = 5 ] || [ $n = 12 ]; then echo "1, 8600"; else echo "1, 560"; fi; }
for n in $(seq 1 21); do
  while IFS=, read -r gi gm; do
    gi=${gi// /}; gm=${gm// /}; [ -n "$gi" ] || continue
    if [ "${gm:-9999}" -lt 3000 ]; then
      eval "gidle$gi=\$(( \${gidle$gi:-0} + 1 ))"
      eval "[ \$gidle$gi = 8 ]" && A WARN "GPU $gi has had no job for 16 min (${gm} MiB in use): idle"
    else eval "gidle$gi=0"; fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
done
echo "final counters: gpu0=$gidle0 gpu1=$gidle1"
