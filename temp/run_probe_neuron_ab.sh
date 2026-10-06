#!/bin/bash
# run_probe_neuron_ab.sh -- 2026-09-30 (Neuro 46, Glenn-approved): 1B B8 FPT vs lif_exact_ml probe on the 5090 (GPU 0),
# run ONLY once the 5090 is free. Interleaved fpt, lifml, fpt, lifml. Records tok/s + GPU-busy % (nvidia-smi utilization.gpu
# = % of time a kernel was running) + peak VRAM. Pre-registered (fused doc §7, cuda-graphs doc §5):
#   lifml >= +10% median tok/s (steps >= 50) vs fpt = speed lever; GPU-busy after switch: <=60% build CUDA-graph Phase 1,
#   >=75% stop. Never touches the 5080.
# WATCH: tail -f /home/glenn/projects/neuro/logs/probe_neuron_ab.log
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/probe_neuron_ab.log
say(){ echo "[nab $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
say "waiting for the 5090: run_5090_seq.sh gone, then GPU0 idle (util 0%) for 5 consecutive min"
while pgrep -f "[r]un_5090_seq.sh" > /dev/null; do sleep 120; done
idle=0
while [ $idle -lt 5 ]; do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0)
  if [ "$u" = "0" ]; then idle=$((idle + 1)); else idle=0; fi
  sleep 60
done
exec 8>logs/5090_seq.lock
until flock -n 8; do say "5090_seq.lock busy (another 5090 sequence started) -- retry in 10 min"; sleep 600; done
say "START: 5090 free, holding logs/5090_seq.lock"
$PY temp/mk_probe_neuron_ab.py >> "$LOG" 2>&1
for run in probe_1b_b8_fpt_r1 probe_1b_b8_lifml_r1 probe_1b_b8_fpt_r2 probe_1b_b8_lifml_r2; do
  rm -rf models/checkpoints/$run
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i 0 -lms 1000 > logs/$run.smi &
  smi=$!
  say "$run START"
  CUDA_VISIBLE_DEVICES=0 $PY src/train_gpu.py --config temp/$run.yaml --device cuda > logs/$run.log 2>&1
  rc=$?
  kill $smi
  rm -rf models/checkpoints/$run
  $PY temp/summarize_probe.py $run $rc | tee -a "$LOG"
done
say "DONE"
