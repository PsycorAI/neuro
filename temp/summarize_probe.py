# summarize_probe.py -- 2026-09-30: one-line summary of a 1B probe run (temp/run_probe_neuron_ab.sh).
# tok/s = median of logged steps >= 50; GPU-busy = median utilization.gpu over the last 45% of nvidia-smi samples
# (compile/autotune is front-loaded, so the tail is steady state); peak = max memory.used (system-wide MiB).
import re, sys, statistics as st
run, rc = sys.argv[1], sys.argv[2]
toks = [int(m.group(2).replace(",", "")) for m in
        (re.match(r"step (\d+) .*\| ([\d,]+) tok/s", l) for l in open(f"/home/glenn/projects/neuro/logs/{run}.log"))
        if m and int(m.group(1)) >= 50]
smi = [tuple(int(v) for v in l.split(",")) for l in open(f"/home/glenn/projects/neuro/logs/{run}.smi") if l.count(",") == 1]
tail = smi[int(len(smi) * 0.55):]
print(f"{run} exit={rc} | median tok/s (steps>=50, n={len(toks)}) {st.median(toks) if toks else 'NA'} | "
      f"GPU-busy median {st.median(u for u, _ in tail) if tail else 'NA'}% | peak MiB {max((m for _, m in smi), default='NA')}")
