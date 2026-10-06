#!/bin/bash
# 2026-09-30: why are random 2 KB reads of train.bin ~34 ms on a SATA SSD? memguard firing? cold vs warm? raw device latency?
cd /home/glenn/projects/neuro
echo "memguard threshold line: $(grep thresh_kb= /usr/local/sbin/drop_caches_if_low.sh | head -1) | metric: $(grep -o 'MemFree\|MemAvailable' /usr/local/sbin/drop_caches_if_low.sh | head -1)"
echo "memguard fires in the last 60 min: $(journalctl -t memguard --since '-60min' --no-pager 2>/dev/null | grep -c dropped)"
grep -E 'MemFree|^Cached|Dirty' /proc/meminfo
/home/glenn/projects/bdh/venv/bin/python - <<'EOF'
import time, numpy as np
d = np.memmap("/home/glenn/projects/bdh/data/tokenized/train.bin", dtype=np.uint32, mode="r")
rng = np.random.default_rng(int(time.time())); st = rng.integers(0, len(d) - 600, size=64)
for label in ("cold", "warm (same offsets)"):
    t = time.time(); lat = []
    for s in st:
        t0 = time.perf_counter(); np.asarray(d[s:s + 513]).sum(); lat.append(time.perf_counter() - t0)
    lat = np.array(lat) * 1000
    print(f"{label:22s} per-read ms: median {np.median(lat):6.2f}  p90 {np.percentile(lat, 90):6.2f}  max {lat.max():6.1f}")
EOF
cat /sys/block/sdd/queue/read_ahead_kb /sys/block/sdd/queue/rotational 2>/dev/null | tr '\n' ' '; echo "(read_ahead_kb rotational)"
