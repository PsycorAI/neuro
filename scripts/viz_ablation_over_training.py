"""Build the ablation-Δ-across-training chart from a Run B log file.

Reads logs_phase25_b.txt, pairs each [eval] line with the preceding step line
to get (tokens_seen, val_ppl, ablated_ppl), and plots:
  (a) val ppl vs ablated ppl as two curves
  (b) the Δ between them, shaded

Saves to assets/ablation_over_training.png. Blog asset.
"""
import os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
LOG = os.path.join(ROOT, "logs_phase25_b.txt")
OUT = os.path.join(ROOT, "assets", "ablation_over_training.png")

STEP_RE = re.compile(r"step\s+(\d+)\s*\|\s*([\d.]+)M tok")
EVAL_RE = re.compile(
    r"\[eval\]\s+val_ppl\s+([\d.]+)\s*\|\s*ablated_ppl\s+([\d.]+)\s*\(memory\s+Δ=\+([\d.]+)\)"
)


def parse_log(path):
    pts = []
    last_tokens = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                last_tokens = float(m.group(2))
                continue
            m = EVAL_RE.search(line)
            if m and last_tokens is not None:
                ppl, abl, delta = map(float, m.groups())
                pts.append((last_tokens, ppl, abl, delta))
    return pts


def main():
    pts = parse_log(LOG)
    if not pts:
        # Run B's log was truncated by an earlier launch-script `>` redirect
        # (since fixed to `>>`). Fall back to the eval points reconstructed
        # from the chat history, plus the final eval_phase25.py measurement.
        # Each tuple: (tokens_M, val_ppl, ablated_ppl, delta).
        pts = [
            ( 32.8, 271.83, 430.04, 158.20),
            (229.4, 143.95, 244.14, 100.19),
            (262.1, 153.88, 286.34, 132.46),
            (294.9, 121.99, 215.12,  93.13),
            (311.0, 128.47, 285.91, 157.44),
            (326.0, 110.43, 216.45, 106.02),
            (344.0, 113.72, 199.66,  85.94),
            (362.1, 103.88, 194.45,  90.57),
            (393.2, 113.61, 198.85,  85.24),
            (426.0,  99.33, 185.57,  86.24),
            (655.4,  89.40, 177.57,  88.17),
            (688.1,  94.79, 177.70,  82.91),
            (700.0,  89.10, 176.34,  87.24),   # final eval_phase25.py @ 30 batches
        ]
        print(f"using {len(pts)} reconstructed eval points (log truncated by old launcher)")
    else:
        print(f"parsed {len(pts)} eval points; spans {pts[0][0]:.1f}M to {pts[-1][0]:.1f}M tokens")

    tok = np.array([p[0] for p in pts])
    ppl = np.array([p[1] for p in pts])
    abl = np.array([p[2] for p in pts])
    delta = np.array([p[3] for p in pts])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    # left: two perplexity curves with shaded delta
    ax[0].fill_between(tok, ppl, abl, alpha=0.18, color="#b5651d", label="memory ablation gap")
    ax[0].plot(tok, abl, "-o", ms=3, color="#bdbdbd", label="ppl with M zeroed")
    ax[0].plot(tok, ppl, "-o", ms=3, color="#b5651d", label="ppl with synaptic memory")
    ax[0].set_xlabel("training tokens (millions)")
    ax[0].set_ylabel("validation perplexity")
    ax[0].set_title("PsycorNeuro 20M: ablation gap stays large throughout training")
    ax[0].legend(loc="upper right", fontsize=9)
    ax[0].grid(True, alpha=0.25)

    # right: delta over training
    ax[1].plot(tok, delta, "-o", ms=4, color="#2a6f97")
    ax[1].set_xlabel("training tokens (millions)")
    ax[1].set_ylabel("Δ perplexity (ablated − with memory)")
    ax[1].set_title("the memory never becomes vestigial")
    ax[1].axhline(0, color="gray", lw=0.5)
    ax[1].grid(True, alpha=0.25)
    # annotate first and last points
    ax[1].annotate(f"+{delta[0]:.0f}", (tok[0], delta[0]),
                   xytext=(5, -12), textcoords="offset points", fontsize=9)
    ax[1].annotate(f"+{delta[-1]:.0f}", (tok[-1], delta[-1]),
                   xytext=(5, 5), textcoords="offset points", fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"saved -> {OUT}")
    print(f"  first eval (tok {tok[0]:.1f}M): ppl {ppl[0]:.2f} -> ablated {abl[0]:.2f} (Δ +{delta[0]:.2f})")
    print(f"  last  eval (tok {tok[-1]:.1f}M): ppl {ppl[-1]:.2f} -> ablated {abl[-1]:.2f} (Δ +{delta[-1]:.2f})")
    print(f"  mean Δ = {delta.mean():.2f}, min = {delta.min():.2f}, max = {delta.max():.2f}")


if __name__ == "__main__":
    main()
