"""Concept → synapse traceability tool.

Visualizes WHICH synaptic entries in the Hebbian memory M store the binding
for a given concept. For each (cue, target) pair, we compute the change
ΔM = M_after_pair - M_before_pair and render it as a heatmap.

The "synapses" are entries of M (a d_mem x d_mem matrix per layer). The visual
question: is the storage *interpretable* — i.e., does each concept land in a
specific, sparse set of synaptic entries that we can point to?

Output: assets/concept_to_synapse_trace.png (4-panel figure)
  Panel 1: ΔM for concept 1
  Panel 2: ΔM for concept 2
  Panel 3: ΔM after both concepts (composition)
  Panel 4: Per-entry overlap = |ΔM_1 * ΔM_2| (where do the two concepts share synapses?)

Usage:
    python scripts/concept_to_synapse_trace.py
"""
import os, sys, glob, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM

ROOT = os.path.join(os.path.dirname(__file__), "..")
CK_DIR = os.path.join(ROOT, "models", "checkpoints", "phase3_5090")
OUT = os.path.join(ROOT, "assets", "concept_to_synapse_trace.png")


def latest(d):
    return sorted(glob.glob(os.path.join(d, "step_*.pt")))[-1]


def build_from_ckpt(device):
    ckpt = latest(CK_DIR)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    c = blob["cfg"]
    m = SpikingHebbianLM(c["vocab"], d=c["d"], n_neurons=c["n_neurons"],
                         d_mem=c["d_mem"],
                         recurrent=c.get("recurrent", False),
                         rec_density=c.get("rec_density", 0.05),
                         compile_safe=c.get("compile", False),
                         tie_weights=c.get("tie_weights", False),
                         n_layers=c.get("n_layers", 1),
                         use_fpt=c.get("use_fpt", False),
                         fpt_K=c.get("fpt_K", 10)).to(device)
    m.load_state_dict(blob["model"], strict=False)
    m.eval()
    return m, c


@torch.no_grad()
def expose_to_pair(model, cue, target, repeats, device):
    """Run model on a [cue,target,cue,target,...] sequence, return final state."""
    seq = [cue, target] * repeats
    x = torch.tensor(seq, device=device).unsqueeze(0)
    _, state = model(x, return_final_state=True)
    return state


def get_M(state, layer):
    M = state["layers"][layer]["M"].detach().cpu().float().numpy()
    if M.ndim == 3:
        M = M[0]
    return M


def heatmap(ax, mat, title, vmax=None):
    if vmax is None:
        vmax = np.abs(mat).max()
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("synapse (key dim)")
    ax.set_ylabel("synapse (value dim)")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0, help="which layer's M to visualize")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    m, c = build_from_ckpt(device)
    print(f"model: {c['n_layers']}-layer spiking, d_mem={c['d_mem']}")
    print(f"showing layer {args.layer}")

    # Two concepts.
    pair_1 = (1234, 8765)
    pair_2 = (3456, 10234)

    state_1 = expose_to_pair(m, pair_1[0], pair_1[1], args.repeats, device)
    state_2 = expose_to_pair(m, pair_2[0], pair_2[1], args.repeats, device)

    # "Composed" state: expose to pair 1 then pair 2 in one sequence.
    seq = [pair_1[0], pair_1[1]] * args.repeats + [pair_2[0], pair_2[1]] * args.repeats
    x = torch.tensor(seq, device=device).unsqueeze(0)
    with torch.no_grad():
        _, state_both = m(x, return_final_state=True)

    M1 = get_M(state_1, args.layer)
    M2 = get_M(state_2, args.layer)
    M_both = get_M(state_both, args.layer)
    overlap = np.abs(M1) * np.abs(M2)

    # Stats
    def sparsity(M, thresh_frac=0.01):
        thresh = np.abs(M).max() * thresh_frac
        return (np.abs(M) > thresh).mean() * 100

    print(f"\n=== synapse stats (layer {args.layer}) ===")
    print(f"  concept 1 ({pair_1[0]}->{pair_1[1]}):  "
          f"|M|max={np.abs(M1).max():.3f}  "
          f"active>1% of max: {sparsity(M1):.2f}%")
    print(f"  concept 2 ({pair_2[0]}->{pair_2[1]}):  "
          f"|M|max={np.abs(M2).max():.3f}  "
          f"active>1% of max: {sparsity(M2):.2f}%")
    print(f"  overlap |M1*M2|max={overlap.max():.4f}  "
          f"correlation: {np.corrcoef(M1.flatten(), M2.flatten())[0,1]:+.3f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    vmax = max(np.abs(M1).max(), np.abs(M2).max(), np.abs(M_both).max())
    im1 = heatmap(axes[0, 0], M1, f"concept 1: token {pair_1[0]} → {pair_1[1]}", vmax)
    im2 = heatmap(axes[0, 1], M2, f"concept 2: token {pair_2[0]} → {pair_2[1]}", vmax)
    im3 = heatmap(axes[1, 0], M_both, "after both concepts (composed)", vmax)
    im4 = axes[1, 1].imshow(overlap, cmap="hot",
                             aspect="auto", interpolation="nearest")
    axes[1, 1].set_title("synapse overlap |M1 × M2| (hot = shared)", fontsize=10)
    axes[1, 1].set_xlabel("synapse (key dim)")
    axes[1, 1].set_ylabel("synapse (value dim)")

    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    fig.suptitle(f"Concept → synapse trace (layer {args.layer}, d_mem={c['d_mem']})",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
