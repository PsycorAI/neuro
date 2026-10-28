"""gpu_meter.py -- 2026-10-25 (Neuro 78, Demo plan v2 W4: the "measured on this GPU" meter). The demo server's GPU
energy, MEASURED:
  * a sampler in its OWN process (no GIL bias: a thread of the busy server woke mostly during CUDA syncs = high
    power, 1.2-1.7x too high) reads NVML instant power every 20 ms (+ the energy counter) into a 15-minute ring;
  * the instant-power integral is the primary number: a wall-plug check (KP115, temp/n78_cal_run.sh) found
    d(wall) / d(instant) = 1.04-1.10 on both cards (= the PSU), while the NVML energy counter reads ~0.80x under steady
    load and lags in short windows (logs/n78_cal_result.txt) -> the counter is reported as a secondary figure only;
  * the server asks for the energy of time segments (prefill / decode / brain-view replay ...) after each turn.
Client: m = GpuMeter(); m.energy([(t0, t1), ...], idle_window=(ta, tb)) -> dict. Sampler: python gpu_meter.py <nvml index>."""
import os, sys, json, time, threading, subprocess

# n79 10-25: the 5080's "instant" power field is a ~100 ms block average that shows a load step ~0.09-0.1 s late
# (temp/n79_power_lag.py, logs/n79_power_lag.log; Part-time Power, arXiv 2312.02741: RTX 3090 likewise). Unshifted, a
# request's integral missed its last ~0.1 s of high power (~15 J of a 60 J answer) and took 0.1 s of idle before it.
LAG = 0.095


def _sampler(idx):
    import pynvml as N
    N.nvmlInit(); h = N.nvmlDeviceGetHandleByIndex(idx); fi = N.NVML_FI_DEV_POWER_INSTANT
    ring, lock = [], threading.Lock()

    def run():
        while True:
            # n82 10-26: a transient NVML error ("The operating system has blocked the request", WSL) once killed this
            # thread at server start -> a frozen ring: every answer = one stale power x its duration, idle 0 W. Skip it.
            try:
                v = N.nvmlDeviceGetFieldValues(h, [fi])[0]
                ok = v.nvmlReturn == 0
                if ok:
                    s = (time.time() - LAG, v.value.uiVal / 1000.0, N.nvmlDeviceGetTotalEnergyConsumption(h))
            except N.NVMLError:
                ok = False
            if ok:
                with lock:
                    ring.append(s)
                    if len(ring) > 45000:                    # ~15 min at 20 ms
                        del ring[:5000]
            time.sleep(0.02)

    threading.Thread(target=run, daemon=True).start()
    print(json.dumps({"ready": True, "name": N.nvmlDeviceGetName(h)}), flush=True)
    for line in sys.stdin:
        q = json.loads(line)
        time.sleep(0.06 + LAG + 0.1)                         # the last samples of the newest segment (+ lag + 1 update)
        with lock:
            pts = list(ring)
        out = {"J": [], "Jc": [], "s": []}
        T = [x[0] for x in pts]
        import bisect

        def pw(t):                                           # power at t, linear between the neighbouring samples
            i = bisect.bisect_left(T, t)
            if i <= 0:
                return pts[0][1]
            if i >= len(pts):
                return pts[-1][1]
            (ta, pa, _), (tb, pb, _) = pts[i - 1], pts[i]
            return pa + (pb - pa) * (t - ta) / (tb - ta) if tb > ta else pb
        for t0, t1 in q["segs"]:
            # the integral of the interpolated power over [t0, t1] (a 5 ms decode step holds no 20 ms sample of its own:
            # n78 test, 10-25, the first version summed only samples inside a segment and read such steps as 0 J)
            i0, i1 = bisect.bisect_right(T, t0), bisect.bisect_left(T, t1)
            xs = [(t0, pw(t0))] + [(pts[i][0], pts[i][1]) for i in range(i0, i1)] + [(t1, pw(t1))]
            out["J"].append(sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(xs, xs[1:])) if pts else 0.0)
            p = pts[i0:i1]
            out["Jc"].append((p[-1][2] - p[0][2]) / 1000.0 if len(p) > 1 else 0.0)
            out["s"].append(t1 - t0)
        ia, ib = q.get("idle", (0, 0))
        w = sorted(x[1] for x in pts if ia <= x[0] <= ib)
        out["idle_W"] = w[len(w) // 10] if w else None       # 10th percentile: robust to a stray burst
        print(json.dumps(out), flush=True)


class GpuMeter:
    def __init__(self, nvml_index):
        self.p = subprocess.Popen([sys.executable, os.path.abspath(__file__), str(nvml_index)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.name = json.loads(self.p.stdout.readline())["name"]
        self.lock = threading.Lock()

    def energy(self, segs, idle_window):
        with self.lock:
            self.p.stdin.write(json.dumps({"segs": segs, "idle": idle_window}) + "\n"); self.p.stdin.flush()
            return json.loads(self.p.stdout.readline())


def load_table(path, gpu_name):
    """The measured v4 / v4f reference for this GPU (gpu_energy_table.json, built by temp/n78_make_table.py from the
    W1 harness: same harness, same card, decode at contexts 512 .. 32k), or None."""
    try:
        t = json.load(open(path))
    except (OSError, ValueError):
        return None
    g = t.get("gpus", {}).get(gpu_name)
    return dict(g, source=t.get("source", ""), v4f=t.get("v4f")) if g else None


def at_context(rows, ctx):
    """Interpolate a {context: value} table at ctx (linear in log-context, clamped at the ends)."""
    import math
    xs = sorted(int(k) for k in rows)
    if ctx <= xs[0]:
        return rows[str(xs[0])]
    if ctx >= xs[-1]:
        return rows[str(xs[-1])]
    for a, b in zip(xs, xs[1:]):
        if a <= ctx <= b:
            f = (math.log(ctx) - math.log(a)) / (math.log(b) - math.log(a))
            ra, rb = rows[str(a)], rows[str(b)]
            return {k: ra[k] + f * (rb[k] - ra[k]) for k in ra}


def turn_report(meter, phases, t0, t1, table):
    """The "gpu_energy" event of one request: energy per segment kind (prefill = prompt processing, decode = one token
    per step, viz = the brain-view replay pass, other = host / tools), total and above the GPU's idle power (measured in
    the 30 s before the request), per-token figures, and the v4 / v4f reference of the same harness at this context."""
    kinds = ("prefill", "decode", "viz")
    # consecutive segments of one kind are merged into one span (the host work between two decode steps -- grammar mask,
    # sampling -- is part of generating): prefill | decode ... decode | viz | (tool) | prefill | decode ... | viz
    spans = []
    for k, a, b, n, c in phases:
        if k not in kinds:
            continue
        if spans and spans[-1][0] == k:
            spans[-1][2], spans[-1][3] = b, spans[-1][3] + n
        else:
            spans.append([k, a, b, n])
    r = meter.energy([(t0, t1)] + [(a, b) for k, a, b, n in spans], (t0 - 30, t0 - 0.3))
    idle = r["idle_W"] if r["idle_W"] is not None else 0.0
    tot_J, tot_s = r["J"][0], t1 - t0
    by = {k: dict(J=0.0, s=0.0, n=0) for k in kinds}
    for (k, a, b, n), j, s in zip(spans, r["J"][1:], r["s"][1:]):
        by[k]["J"] += j; by[k]["s"] += s; by[k]["n"] += n
    for k in kinds:
        by[k]["J_active"] = by[k]["J"] - idle * by[k]["s"]
    other_s = max(tot_s - sum(by[k]["s"] for k in kinds), 0.0)
    by["other"] = dict(J=max(tot_J - sum(by[k]["J"] for k in kinds), 0.0), s=other_s)
    by["other"]["J_active"] = by["other"]["J"] - idle * other_s
    dec = by["decode"]
    ctx = max([c for k, a, b, n, c in phases] or [0])
    first = [b for k, a, b, n, c in phases if k == "decode"]
    ev = dict(type="gpu_energy", gpu=meter.name, secs=tot_s, idle_W=idle, total_J=tot_J, total_J_active=tot_J - idle * tot_s,
              counter_J=r["Jc"][0], by=by, context=ctx,
              model_J=by["prefill"]["J"] + dec["J"], model_J_active=by["prefill"]["J_active"] + dec["J_active"],
              decode_tokens=dec["n"], tok_s=dec["n"] / dec["s"] if dec["s"] else None,
              mJ_tok=1e3 * dec["J"] / dec["n"] if dec["n"] else None,
              mJ_tok_active=1e3 * dec["J_active"] / dec["n"] if dec["n"] else None,
              first_token_s=(first[0] - t0) if first else None)
    if table and dec["n"]:
        v4, v4f = at_context(table["v4_dec"], ctx), at_context(table["v4f_dec"], ctx)
        ev["ref"] = dict(context=ctx, v4=v4, v4f=v4f, curve=dict(v4=table["v4_dec"], v4f=table["v4f_dec"]),
                         source=table.get("source", ""))
    return ev


def nvml_index_of_cuda0():
    """NVML index of the GPU this process sees as cuda:0 (matched by UUID; NVML ignores CUDA_VISIBLE_DEVICES)."""
    import torch, pynvml as N
    N.nvmlInit()
    u = str(torch.cuda.get_device_properties(0).uuid).replace("GPU-", "")
    for i in range(N.nvmlDeviceGetCount()):
        x = N.nvmlDeviceGetUUID(N.nvmlDeviceGetHandleByIndex(i)); x = x.decode() if isinstance(x, bytes) else x
        if x.replace("GPU-", "") == u:
            return i
    raise RuntimeError("no NVML device matches cuda:0")


if __name__ == "__main__":
    _sampler(int(sys.argv[1]))
