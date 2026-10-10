#!/bin/bash
# reorder_1540.sh -- 2026-10-07 15:40 (Neuro 53). HOLA first (Glenn). Stop only WAITING queue scripts; running jobs
# (bench C last step, p26y sched/train) continue. Then relaunch one chain per GPU.
cd /home/glenn/projects/neuro
k(){ for p in $(pgrep -f "$1"); do echo "TERM $p: $(tr '\0' ' ' < /proc/$p/cmdline)"; kill -TERM $p; done; }
k "^bash temp/queue_5090_r26g.sh$"
# queue_sc.sh 5090 branch = the queue_sc.sh subshell that is NOT the parent of the p26y scheduler
P26Y=$(pgrep -f "^bash temp/run_5080_sched.sh screen_p26y_7L_chunk_rand128$")
KEEP=$(ps -o ppid= -p $P26Y | tr -d ' ')
for p in $(pgrep -f "^bash temp/queue_sc.sh$"); do
  pp=$(ps -o ppid= -p $p | tr -d ' ')
  if [ "$p" != "$KEEP" ] && [ "$pp" != "1" ] && [ "$p" != "$(ps -o ppid= -p $KEEP | tr -d ' ')" ]; then echo "TERM queue_sc 5090 branch $p"; kill -TERM $p; fi
done
k "^bash temp/queue_5090_rc.sh$"
sleep 2
echo "--- remaining:"; ps -eo pid,ppid,args | grep -E 'queue|sched|bench|train_gpu' | grep -v grep | grep -v reorder | cut -c1-150
/home/glenn/projects/bdh/venv/bin/python temp/mk_p26_configs.py | grep -E 'p26z|r26h'
