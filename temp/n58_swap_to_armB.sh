#!/bin/bash
# Neuro 58: probe A has answered (cross-chunk recall 93-100% after 200 steps). Stop its trainer (SIGTERM, exact
# pattern), then start arm B (embedding decay off, configs/probe_v4_nodecay.yaml) with its own eval follower.
cd /home/glenn/projects/neuro || exit 1
pkill -TERM -f 'src/train_gpu.py --config configs/probe_v4_xchunk.yaml'
for i in $(seq 1 60); do pgrep -f 'src/train_gpu.py --config configs/probe_v4_xchunk.yaml' > /dev/null || break; sleep 2; done
pgrep -af 'src/train_gpu.py --config configs/probe_v4_xchunk.yaml' | cut -c1-80 && { echo "trainer A still alive"; exit 1; }
setsid -f bash temp/n58_probe.sh probe_v4_nodecay < /dev/null > /dev/null 2>&1
setsid -f bash temp/n58_probe_follow.sh probe_v4_nodecay < /dev/null > /dev/null 2>&1
sleep 5; pgrep -af '^bash temp/(n58_|queue_5090_n58hold|watchdog_tps)' | cut -c1-80
ls -la models/checkpoints/probe_v4_xchunk/ | tail -n 3
