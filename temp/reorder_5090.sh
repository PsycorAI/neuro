#!/bin/bash
# reorder_5090.sh -- 2026-10-11 (Neuro 55): run r26z (tokens-per-step test, launch-relevant) BEFORE r26y (conversion
# control, not launch-relevant). r26y started at 15:03 and has no checkpoint of its own yet, so stopping it loses
# ~15 minutes. Stops ONLY the 5090 chain (queue script, its scheduler instance for r26y, r26y's trainer), by exact
# command lines; puts r26z ahead of r26y in the list; takes r26y off the done-list; restarts the chain.
cd /home/glenn/projects/neuro
[ -f configs/r26z_13L_RC3_tok32k_150m.yaml ] || { echo "r26z config missing"; exit 1; }
pgrep -f "src/train_gpu.py --config configs/r26y_13L_RC3_kdonly_150m.yaml" > /dev/null || { echo "r26y is not training, NOT touching the chain"; exit 1; }
q=$(pgrep -f "^bash temp/queue_5090_n54.sh"); s=$(pgrep -f "^bash temp/run_5080_sched.sh r26y_13L_RC3_kdonly_150m"); t=$(pgrep -f "src/train_gpu.py --config configs/r26y_13L_RC3_kdonly_150m.yaml")
echo "chain $q | scheduler $s | trainer $t"
[ -n "$q" ] && kill -TERM $q; sleep 1
[ -n "$s" ] && kill -TERM $s; sleep 1
[ -n "$t" ] && kill -TERM $t
for i in $(seq 1 30); do pgrep -f "src/train_gpu.py --config configs/r26y_13L_RC3_kdonly_150m.yaml" > /dev/null || break; sleep 2; done
pgrep -f "src/train_gpu.py --config configs/r26y_|^bash temp/queue_5090_n54.sh|^bash temp/run_5080_sched.sh r26y_" > /dev/null && { echo "something of the 5090 chain is still alive, NOT restarting"; exit 1; }
# r26y back to its seed only (drop any checkpoint written after the seed; the seed step is 12208)
ls models/checkpoints/r26y_13L_RC3_kdonly_150m/
sed -i '/^r26y_13L_RC3_kdonly_150m$/d' temp/queue_5090_n54.done
sed -i -E 's/^r26y_13L_RC3_kdonly_150m$/r26z_13L_RC3_tok32k_150m\nr26y_13L_RC3_kdonly_150m/' temp/queue_5090_n54.list
echo "[n55 $(date '+%m-%d %H:%M:%S')] 5090 reorder: r26y stopped (restarts from its seed later), r26z first" >> logs/gap_queue.log
setsid -f bash temp/queue_5090_n54.sh < /dev/null > /dev/null 2>&1
sleep 3; tail -n 3 temp/queue_5090_n54.list; pgrep -af "^bash temp/queue_5090|run_5080_sched.sh r26" | cut -c1-100
