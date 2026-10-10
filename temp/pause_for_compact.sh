#!/bin/bash
# pause_for_compact.sh -- 2026-10-07 22:10 (Neuro 53). Glenn compacts the WSL vhdx: stop queues + wrappers, then SIGINT
# both trainers (finish step, save final ckpt, exit 130). Resume later with temp/resume_after_compact.sh.
cd /home/glenn/projects/neuro
k(){ for p in $(pgrep -f "$1"); do echo "$2 $p: $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-90)"; kill -$2 $p; done; }
k "^bash temp/queue_5090_chain2.sh$" TERM
k "^bash temp/queue_5080_chain2.sh$" TERM
k "^bash temp/watchdog_tps.sh$" TERM
k "^bash temp/run_gap_queue.sh r26h_13L_RC_hola64_150m$" TERM
k "^bash temp/run_5080_sched.sh screen_p26z_7L_hola64$" TERM
sleep 1
k "train_gpu.py --config configs/r26h_13L_RC_hola64_150m.yaml" INT
k "train_gpu.py --config configs/screen_p26z_7L_hola64.yaml" INT
for i in $(seq 1 90); do pgrep -f "src/train_gpu.py" > /dev/null || break; sleep 2; done
pgrep -af "src/train_gpu.py" && echo "STILL RUNNING" || echo "both trainers exited"
ls models/checkpoints/r26h_13L_RC_hola64_150m/ models/checkpoints/screen_p26z_7L_hola64/
echo "[n53 $(date '+%m-%d %H:%M:%S')] PAUSED for vhdx compaction (Glenn): r26h + p26z SIGINT -> final ckpts; resume with temp/resume_after_compact.sh" >> logs/gap_queue.log
ps -eo pid,args | grep -E "queue|sched|run_gap|train_gpu|watchdog" | grep -v grep | grep -v pause_for | cut -c1-120
