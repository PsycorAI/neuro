"""ST Phase 1.4 — stateful eval (2x2 + ppl-vs-context-position curve).

Evaluates one or more trained checkpoints under TWO eval regimes on a long
CONTIGUOUS held-out slice of the validation tail:

  * zero-state  : M (and LIF state) reset at every block boundary (cold start).
  * stateful    : state carried across blocks (detached), like real streaming use.

For each (run, regime) we report perplexity over the whole slice. We also bin
per-chunk perplexity by context position to produce the money-shot curve:
a stateful-trained model should get BETTER as carried context accumulates; a
random-trained model carried-state-evaluated may get WORSE (off-distribution M
= the brain-mod saturation effect).

Usage:
  # default: the two ST Phase 1.3 checkpoints
  python scripts/stateful_eval.py
  # smoke / custom:
  python scripts/stateful_eval.py --runs phase3_5090 --eval_tokens 50000
"""
import os, sys, glob, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM

ROOT = os.path.join(os.path.dirname(__file__), "..")
BIN = "/home/glenn/projects/bdh/data/tokenized/train.bin"
OUT = os.path.join(ROOT, "assets", "stateful_eval_curve.png")


def latest(run):
    d = os.path.join(ROOT, "models", "checkpoints", run)
    cks = sorted(glob.glob(os.path.join(d, "step_*.pt")))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {d}")
    return cks[-1]


def build(c, device):
    m = SpikingHebbianLM(c["vocab"], d=c["d"], n_neurons=c["n_neurons"],
                         d_mem=c["d_mem"],
                         recurrent=c.get("recurrent", False),
                         rec_density=c.get("rec_density", 0.05),
                         compile_safe=c.get("compile", False),
                         tie_weights=c.get("tie_weights", False),
                         n_layers=c.get("n_layers", 1),
                         use_fpt=c.get("use_fpt", False),
                         fpt_K=c.get("fpt_K", 10),
                         lam=c.get("lam", 0.98),
                         learnable_decay=c.get("learnable_decay", False),
                         write_gate=c.get("write_gate", False),
                         delta_rule=c.get("delta_rule", False)).to(device)
    return m


def contiguous_val_slice(vocab, n_tokens):
    """A single contiguous slice from the validation tail, remapped to student vocab."""
    data = np.memmap(BIN, dtype=np.uint32, mode="r")
    lut = np.load(f"/home/glenn/projects/neuro/data/vocab_map_{vocab}.npy")
    val_lo = int(len(data) * 0.999)
    n_tokens = min(n_tokens, len(data) - val_lo - 1)
    raw = np.asarray(data[val_lo:val_lo + n_tokens + 1], dtype=np.int64)
    return lut[raw]


@torch.no_grad()
def eval_stream(model, tokens, block, device, stateful):
    """Return (overall_ppl, per_chunk_ppl list). One contiguous pass, B=1."""
    model.eval()
    n_chunks = (len(tokens) - 1) // block
    carry = None
    total_ce, total_tok = 0.0, 0
    per_chunk = []
    for i in range(n_chunks):
        s = i * block
        x = torch.from_numpy(tokens[s:s + block][None]).to(device)
        y = torch.from_numpy(tokens[s + 1:s + 1 + block][None]).to(device)
        if stateful:
            out, carry = model(x, initial_state=carry, return_final_state=True)
        else:
            out = model(x)
        ce = F.cross_entropy(out.reshape(-1, model.vocab), y.reshape(-1))
        per_chunk.append(math.exp(ce.item()))
        total_ce += ce.item() * y.numel()
        total_tok += y.numel()
    return math.exp(total_ce / total_tok), per_chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="phase25_st_random,phase25_st_stateful",
                    help="comma-separated run_names to evaluate")
    ap.add_argument("--eval_tokens", type=int, default=300000)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device
    runs = args.runs.split(",")

    # Load all runs' configs/weights up front (assume shared vocab/block-compatible)
    models, cfgs = {}, {}
    for run in runs:
        blob = torch.load(latest(run), map_location=device, weights_only=False)
        c = blob["cfg"]
        m = build(c, device)
        missing, unexpected = m.load_state_dict(blob["model"], strict=False)
        dropped = [k for k in unexpected if "lif." not in k]  # lif buffers are expected
        if dropped:
            print(f"  [WARN] {run}: ckpt has keys the built model lacks "
                  f"(arch mismatch?): {dropped[:4]}")
        if missing:
            print(f"  [WARN] {run}: model expects keys absent from ckpt: {missing[:4]}")
        models[run] = m
        cfgs[run] = c
        print(f"loaded {run}: {sum(p.numel() for p in m.parameters())/1e6:.2f}M params, "
              f"step {blob.get('step')}")

    vocab = cfgs[runs[0]]["vocab"]
    tokens = contiguous_val_slice(vocab, args.eval_tokens)
    print(f"eval slice: {len(tokens):,} contiguous val-tail tokens, block={args.block}\n")

    # 2x2 (or Nx2) table
    print(f"{'run':>26} | {'zero-state ppl':>15} | {'stateful ppl':>13} | {'Δ%':>7}")
    print("-" * 72)
    curves = {}
    for run in runs:
        z_ppl, z_curve = eval_stream(models[run], tokens, args.block, device, stateful=False)
        s_ppl, s_curve = eval_stream(models[run], tokens, args.block, device, stateful=True)
        delta = (z_ppl - s_ppl) / z_ppl * 100
        print(f"{run:>26} | {z_ppl:>15.2f} | {s_ppl:>13.2f} | {delta:>+6.1f}%")
        curves[run] = s_curve

    # Curve: stateful per-chunk ppl vs context position
    plt.figure(figsize=(8, 5))
    for run in runs:
        y = curves[run]
        x = [(i + 1) * args.block for i in range(len(y))]
        label = run.replace("phase25_st_", "")
        plt.plot(x, y, marker=".", ms=3, lw=1, label=label)
    plt.xlabel("context position (tokens of carried state)")
    plt.ylabel("per-chunk perplexity (stateful eval)")
    plt.title("Does perplexity improve as context accumulates?")
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=140)
    print(f"\nsaved curve -> {OUT}")
    print("Reading: downward slope = model uses accumulated context (good).")
    print("         flat/upward = carried state is off-distribution (saturation).")


if __name__ == "__main__":
    main()
