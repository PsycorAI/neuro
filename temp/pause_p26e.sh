#!/bin/bash
# pause_p26e.sh -- 2026-10-06 20:10 (Glenn needs the 5080 now). SIGINT = finish step, save final ckpt, exit 130.
cd /home/glenn/projects/neuro
for p in $(pgrep -f "train_gpu.py --config configs/screen_p26e_7L_xscatter.yaml"); do
  echo "SIGINT $p: $(tr '\0' ' ' < /proc/$p/cmdline)"; kill -INT $p
done
for i in $(seq 1 60); do pgrep -f "train_gpu.py --config configs/screen_p26e_7L_xscatter.yaml" > /dev/null || break; sleep 2; done
pgrep -af "train_gpu.py --config configs/screen_p26e" && echo "STILL RUNNING" || echo "stopped"
tail -n 3 logs/screen_p26e_7L_xscatter.log
ls models/checkpoints/screen_p26e_7L_xscatter/
