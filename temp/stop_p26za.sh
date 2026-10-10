#!/bin/bash
cd /home/glenn/projects/neuro
for p in $(pgrep -f "^bash temp/queue_5080_p26.sh screen_p26z"); do echo "TERM $p"; kill -TERM $p; done
for p in $(pgrep -f "^bash temp/run_5080_sched.sh screen_p26za"); do echo "TERM $p"; kill -TERM $p; done
sleep 2
for p in $(pgrep -f "src/train_gpu.py"); do echo "STILL TRAINING $p"; done
ps -eo pid,args | grep -E "queue|sched|run_gap|train_gpu|watchdog" | grep -v grep | grep -v stop_p26za | cut -c1-120
ls -d models/checkpoints/*hola* models/checkpoints/r26d* models/checkpoints/r26c* models/checkpoints/screen_p26y* models/checkpoints/screen_r22a* 2>&1
grep -E "save|Error|rror" logs/r26h_13L_RC_hola64_150m.log | tail -5
