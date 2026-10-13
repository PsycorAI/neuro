#!/bin/bash
# n58_cloud_apply.sh stage|check|apply|verify|rollback -- 2026-10-12 (Neuro 58; Glenn OK 10-12 13:30). The
# embedding-decay fix on the v4 cloud run. stage = upload the two src files + the remote script (changes nothing that
# runs); check = read-only state of the instance; apply = run temp/n58_apply_remote.sh there; verify = read the
# restart from the log; rollback = put the launch files and configs back and SIGINT the trainer (same restart path).
cd /home/glenn/projects/neuro || exit 1
. temp/n56_host.env
case "$1" in
  stage)   $SSH root@$H 'mkdir -p /workspace/neuro/n58_stage' 2>/dev/null
           scp -q -P $P -o BatchMode=yes src/muon.py src/train_gpu.py temp/n58_apply_remote.sh root@$H:/workspace/neuro/n58_stage/ \
             && $SSH root@$H 'cd /workspace/neuro/n58_stage && md5sum muon.py train_gpu.py n58_apply_remote.sh' 2>/dev/null
           md5sum src/muon.py src/train_gpu.py temp/n58_apply_remote.sh ;;
  check)   $SSH root@$H 'cd /workspace/neuro; md5sum src/muon.py src/train_gpu.py; pgrep -af "cloud/run_v4.sh|src/train_gpu.py" | cut -c1-110; grep -E "^step|saved" logs/v4_cloud.log | tail -n 3 | cut -c1-110; grep -c . configs/v4_cloud.yaml configs/v4_cloud_conv.yaml; grep -n "weight_decay\|no_decay" configs/v4_cloud.yaml configs/v4_cloud_conv.yaml' 2>/dev/null ;;
  apply)   $SSH root@$H 'cd /workspace/neuro && bash n58_stage/n58_apply_remote.sh' 2>/dev/null ;;
  verify)  $SSH root@$H 'cd /workspace/neuro; tail -n 4 logs/run_v4.log | cut -c1-120; grep -n -E "no_decay|remap|^resumed|warn|^[A-Za-z]*Error|SIGINT" logs/v4_cloud.log | tail -n 8 | cut -c1-200; grep -E "^step|eval\]|\[wnorm\]|saved" logs/v4_cloud.log | tail -n 8 | cut -c1-160; pgrep -af "src/train_gpu.py --config" | cut -c1-90; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader' 2>/dev/null ;;
  rollback) $SSH root@$H 'cd /workspace/neuro && for f in src/muon.py src/train_gpu.py configs/v4_cloud.yaml configs/v4_cloud_conv.yaml; do [ -f $f.pre_n58 ] && cp -p $f.pre_n58 $f; done; md5sum src/muon.py src/train_gpu.py; pid=$(pgrep -f "python src/train_gpu.py --config configs/v4_cloud" | head -n 1); [ -n "$pid" ] && kill -INT $pid && echo "SIGINT -> $pid"' 2>/dev/null ;;
esac
