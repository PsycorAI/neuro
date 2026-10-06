"""trace_cpu.py -- summarize the CPU side of a torch.profiler chrome trace: top-level (depth-0 per
thread) CPU ranges by total duration, plus GPU busy fraction over the profiled window."""
import json, sys, collections
tr = json.load(open(sys.argv[1]))
ev = [e for e in tr["traceEvents"] if e.get("ph") == "X" and "dur" in e]
cpu = [e for e in ev if e.get("cat") in ("cpu_op", "user_annotation", "python_function") and not e["name"].startswith("ProfilerStep")]
gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
t0 = min(e["ts"] for e in ev); t1 = max(e["ts"] + e["dur"] for e in ev)
# GPU busy = union of kernel intervals
iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in gpu); busy = 0; cs = ce = None
for s, e in iv:
    if cs is None or s > ce:
        if cs is not None: busy += ce - cs
        cs, ce = s, e
    else: ce = max(ce, e)
if cs is not None: busy += ce - cs
print(f"window {(t1-t0)/1e6:.2f}s | GPU busy {busy/1e6:.2f}s ({100*busy/(t1-t0):.0f}%) | {len(gpu)} GPU kernels/copies")
# top-level CPU ops per thread (not contained in another op on the same thread)
by_tid = collections.defaultdict(list)
for e in cpu: by_tid[e["tid"]].append(e)
agg = collections.Counter(); cnt = collections.Counter()
for tid, es in by_tid.items():
    es.sort(key=lambda e: (e["ts"], -e["dur"])); end = -1
    for e in es:
        if e["ts"] >= end:
            agg[e["name"][:90]] += e["dur"]; cnt[e["name"][:90]] += 1; end = e["ts"] + e["dur"]
tot = sum(agg.values())
print(f"top-level CPU time {tot/1e6:.2f}s across {len(by_tid)} threads")
for n, d in agg.most_common(30):
    print(f"  {d/1e6:7.3f}s {100*d/tot:5.1f}%  x{cnt[n]:5d}  {n}")
