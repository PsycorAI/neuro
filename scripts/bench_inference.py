"""Inference-throughput benchmark for the 20M spiking and transformer models.

Compares:
  - eager (no compile)
  - torch.compile(mode="default")
  - torch.compile(mode="reduce-overhead")  -- uses CUDA Graphs, big win for
    small fixed-shape inference workloads (NVIDIA's recommended path for LLM
    inference, 2026).

Reports steady-state tok/s at multiple seq lengths."""
import os, sys, glob, time, argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM
from baseline import TinyTransformer

ROOT = os.path.join(os.path.dirname(__file__), "..")
SPK_DIR = os.path.join(ROOT, "models", "checkpoints", "phase25_b")
TF_DIR  = os.path.join(ROOT, "models", "checkpoints", "phase25_b_tf")


def latest(d):
    cks = sorted(glob.glob(os.path.join(d, "step_*.pt")))
    return cks[-1] if cks else None


def build(c):
    if c["arch"] == "spiking":
        return SpikingHebbianLM(c["vocab"], d=c["d"], n_neurons=c["n_neurons"],
                                d_mem=c["d_mem"],
                                recurrent=c.get("recurrent", False),
                                rec_density=c.get("rec_density", 0.05),
                                compile_safe=c.get("compile", False),
                                n_layers=c.get("n_layers", 1))
    return TinyTransformer(c["vocab"], d=c["d"], n_head=c["n_head"],
                           n_layer=c["n_layer"], max_T=c["block_size"])


def load(path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    c = blob["cfg"]
    m = build(c).to(device).eval()
    m.load_state_dict(blob["model"])
    return m, c


@torch.no_grad()
def bench(model, c, seq_len, device, n_iters=30, warmup=10):
    if seq_len > c["block_size"]:
        return None
    x = torch.randint(0, c["vocab"], (1, seq_len), device=device)
    for _ in range(warmup):
        _ = model(x)
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        _ = model(x)
    if device == "cuda": torch.cuda.synchronize()
    return n_iters * seq_len / (time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seq-lens", default="128,512")
    args = ap.parse_args()
    device = args.device
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    seq_lens = [int(s) for s in args.seq_lens.split(",")]

    print(f"device: {device.upper()}")
    for label, cdir in (("SPIKING", SPK_DIR), ("TRANSFORMER", TF_DIR)):
        ck = latest(cdir)
        if not ck:
            print(f"{label}: no checkpoint, skip"); continue
        print()
        print(f"=== {label} : {os.path.basename(ck)} ===")
        m_eager, c = load(ck, device)
        try:
            m_def = torch.compile(load(ck, device)[0], mode="default", dynamic=False)
        except Exception as e:
            m_def = None; print(f"  compile(default) build failed: {e}")
        try:
            m_red = torch.compile(load(ck, device)[0], mode="reduce-overhead", dynamic=False)
        except Exception as e:
            m_red = None; print(f"  compile(reduce-overhead) build failed: {e}")

        print(f"  {'seq_len':>7} | {'eager':>10} | {'compile(default)':>16} | {'compile(reduce-overhead)':>24}")
        print("  " + "-" * 75)
        for L in seq_lens:
            r_eager = bench(m_eager, c, L, device)
            r_def   = bench(m_def, c, L, device) if m_def is not None else None
            r_red   = bench(m_red, c, L, device) if m_red is not None else None
            cells = [
                f"{r_eager:>10,.0f}" if r_eager else f"{'(skip)':>10}",
                f"{r_def:>16,.0f}"   if r_def   else f"{'(skip)':>16}",
                f"{r_red:>24,.0f}"   if r_red   else f"{'(skip)':>24}",
            ]
            print(f"  {L:>7} | {cells[0]} | {cells[1]} | {cells[2]}")
        del m_eager, m_def, m_red
        if device == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
