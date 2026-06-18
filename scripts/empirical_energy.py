"""Empirical inference energy measurements for both 20M models.

Samples real GPU power draw via nvidia-smi while running fixed-length inference,
then computes actual joules per token. This complements the theoretical 45 nm
AC + MAC figures in the existing eval_phase25.py with hardware-realized numbers
that can be cited in Blog #3 without the "neuromorphic hardware would..." caveat.

Usage:
  python scripts/empirical_energy.py
  python scripts/empirical_energy.py --duration 20 --seq-lens 128,512,2048,4096
"""
import os, sys, glob, time, argparse, subprocess, threading, statistics
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM
from baseline import TinyTransformer

ROOT = os.path.join(os.path.dirname(__file__), "..")
SPK_DIR = os.path.join(ROOT, "models", "checkpoints", "phase25_b")
TF_DIR  = os.path.join(ROOT, "models", "checkpoints", "phase25_b_tf")


def latest_ckpt(d):
    cks = sorted(glob.glob(os.path.join(d, "step_*.pt")))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {d}")
    return cks[-1]


def build_from_cfg(c):
    if c["arch"] == "spiking":
        return SpikingHebbianLM(c["vocab"], d=c["d"], n_neurons=c["n_neurons"],
                                d_mem=c["d_mem"],
                                recurrent=c.get("recurrent", False),
                                rec_density=c.get("rec_density", 0.05),
                                compile_safe=c.get("compile", False))
    return TinyTransformer(c["vocab"], d=c["d"], n_head=c["n_head"],
                           n_layer=c["n_layer"], max_T=c["block_size"])


def load(ckpt_path, device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = blob["cfg"]
    m = build_from_cfg(c).to(device).eval()
    m.load_state_dict(blob["model"])
    return m, c


class PowerSampler(threading.Thread):
    """Background thread sampling GPU power.draw every ~50 ms via nvidia-smi."""

    def __init__(self, gpu_index=0, interval=0.05):
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples = []
        self.stop_evt = threading.Event()

    def run(self):
        cmd = ["nvidia-smi",
               f"--id={self.gpu_index}",
               "--query-gpu=power.draw",
               "--format=csv,noheader,nounits"]
        while not self.stop_evt.is_set():
            try:
                out = subprocess.check_output(cmd, timeout=2).decode().strip()
                w = float(out)
                self.samples.append((time.time(), w))
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=2)


def avg_power_between(samples, t_start, t_end):
    inside = [w for (t, w) in samples if t_start <= t <= t_end]
    return statistics.mean(inside) if inside else float("nan")


@torch.no_grad()
def measure(model, c, seq_len, duration_s, device):
    """Run inference at given seq_len for ~duration_s, return (avg_power, tokens)."""
    if seq_len > c["block_size"]:
        return None
    x = torch.randint(0, c["vocab"], (1, seq_len), device=device)

    # warm up so compile/CUDA-graph any first-call overhead is excluded
    for _ in range(5):
        _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    sampler = PowerSampler()
    sampler.start()
    time.sleep(0.5)                        # let sampler accumulate idle samples
    idle_until = time.time()
    t0 = time.time()
    n = 0
    while time.time() - t0 < duration_s:
        _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        n += 1
    t1 = time.time()
    time.sleep(0.3)
    sampler.stop()

    idle_w = avg_power_between(sampler.samples, sampler.samples[0][0], idle_until)
    run_w  = avg_power_between(sampler.samples, t0, t1)
    delta_w = run_w - idle_w if (idle_w == idle_w and run_w == run_w) else float("nan")
    elapsed = t1 - t0
    tokens = n * seq_len
    energy_j = delta_w * elapsed if delta_w == delta_w else float("nan")
    return {
        "seq_len": seq_len,
        "duration_s": elapsed,
        "n_forwards": n,
        "tokens": tokens,
        "idle_w": idle_w,
        "run_w": run_w,
        "delta_w": delta_w,
        "energy_j": energy_j,
        "energy_per_tok_uj": (energy_j * 1e6 / tokens) if delta_w == delta_w else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--duration", type=float, default=15.0,
                    help="seconds of inference per (model, seq_len) cell")
    ap.add_argument("--seq-lens", default="128,512,2048,4096",
                    help="comma-separated seq_lens to test")
    args = ap.parse_args()
    if args.device != "cuda":
        print("Empirical energy only meaningful on CUDA GPU."); sys.exit(1)
    seq_lens = [int(s) for s in args.seq_lens.split(",")]

    spk, sc = load(latest_ckpt(SPK_DIR), args.device)
    tf,  tc = load(latest_ckpt(TF_DIR),  args.device)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Spiking: {sum(p.numel() for p in spk.parameters())/1e6:.2f}M | "
          f"Transformer: {sum(p.numel() for p in tf.parameters())/1e6:.2f}M")
    print(f"Per-cell sampling duration: {args.duration:.0f}s")

    print()
    print(f"{'seq':>6} | {'model':<8} | {'idle W':>7} {'run W':>7} {'ΔW':>6} | "
          f"{'tok':>9} {'sec':>5} | {'µJ/tok':>9} | {'mJ/1k tok':>10}")
    print("-" * 92)

    results = {"spiking": [], "transformer": []}
    for seq_len in seq_lens:
        for label, model, c in (("spiking", spk, sc), ("transf", tf, tc)):
            r = measure(model, c, seq_len, args.duration, args.device)
            if r is None:
                print(f"{seq_len:>6} | {label:<8} | (exceeds model block_size)")
                continue
            results[label if label != "transf" else "transformer"].append(r)
            print(f"{seq_len:>6} | {label:<8} | {r['idle_w']:>6.1f}  {r['run_w']:>6.1f} "
                  f"{r['delta_w']:>6.1f} | {r['tokens']:>9,} {r['duration_s']:>5.1f} | "
                  f"{r['energy_per_tok_uj']:>9.2f} | {r['energy_per_tok_uj']:>10.2f}")

    # comparison
    print("\n=== Spiking vs Transformer (empirical, current dense GPU) ===")
    for sr, tr in zip(results["spiking"], results["transformer"]):
        if sr["seq_len"] != tr["seq_len"]: continue
        ratio = tr["energy_per_tok_uj"] / sr["energy_per_tok_uj"]
        sign = "spiking wins" if ratio > 1 else "transformer wins"
        print(f"  seq {sr['seq_len']:>5}: spk {sr['energy_per_tok_uj']:>7.2f} µJ/tok  "
              f"tf {tr['energy_per_tok_uj']:>7.2f} µJ/tok  →  {ratio:.2f}x ({sign})")


if __name__ == "__main__":
    main()
