#!/bin/bash
# fix_0405.sh -- r26i on the 16 GB 5080: B2xGA4 (same tokens/step) + expandable segments; watchdog threshold 5k tok/s.
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
$PY - << 'PYEOF'
import yaml
p = "configs/r26i_13L_RC2_hola_rand128_150m.yaml"
c = yaml.safe_load(open(p)); hdr = open(p).readline()
c.update(batch_size=2, grad_accum=4, eval_batch=1, mqar_aug_batch=8, mqar_batch=2, mqar_xchunk_batch=2)
open(p, "w").write(hdr.rstrip() + " [10-08 04:05: B2xGA4 for the 16 GB 5080, same tokens/step]\n" + yaml.safe_dump(c, sort_keys=False))
print("r26i:", {k: c[k] for k in ("batch_size", "grad_accum", "eval_batch", "mqar_aug_batch", "mqar_batch", "mqar_xchunk_batch")})
PYEOF
for p in $(pgrep -f "^bash temp/watchdog_tps.sh$"); do kill -TERM $p; done
MIN_TPS=5000 setsid -f bash temp/watchdog_tps.sh > /dev/null 2>&1
rm -rf models/checkpoints/r26i_13L_RC2_hola_rand128_150m
setsid -f bash temp/queue_5080_r26i.sh > /dev/null 2>&1
echo "[n53 $(date '+%m-%d %H:%M:%S')] r26i crashed on the 5080 (CUDA 'device not ready' at B4 = VRAM pressure); retry B2xGA4 + expandable segments; watchdog MIN_TPS 5000" >> logs/gap_queue.log
sleep 2; pgrep -af "watchdog|r26i" | grep -v fix_0405
