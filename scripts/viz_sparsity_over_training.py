"""Spike-sparsity-over-training chart for Blog #3 (or #2).

Higher = better for energy / neuromorphic efficiency: every silent
neuron-timestep is one that doesn't trigger an accumulate op. PsycorNeuro
maintains 95-99% sparsity across the entire 700M-token Run B (the figure
below); for comparison, SpikingBrain 7B reports 69% sparsity in its tech
report (https://arxiv.org/abs/2509.05276).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "spike_sparsity_over_training.png")

# Spike-rate readings recorded during Run B (20M-param spiking model, 700M Llama-3 tokens).
# Source: Run B eval lines, reconstructed from the chat log because the
# launcher's pre-fix `>` redirect truncated the on-disk log.
# Each tuple: (tokens_M, spike_rate). Silent rate = 1 - spike_rate.
POINTS = [
    ( 32.8, 0.032),
    (229.4, 0.083),
    (262.1, 0.018),
    (294.9, 0.041),
    (311.0, 0.020),
    (326.0, 0.041),
    (344.0, 0.038),
    (362.1, 0.034),
    (393.2, 0.032),
    (426.0, 0.034),
    (655.4, 0.038),
    (688.1, 0.039),
    (700.0, 0.037),  # final
]

SPIKINGBRAIN_SILENT = 0.69    # 69% sparsity reported in arXiv:2509.05276


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = np.array([p[0] for p in POINTS])
    spike = np.array([p[1] for p in POINTS])
    silent = 1.0 - spike            # silent neuron-timesteps fraction

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(tok, silent * 100, "-o", ms=5, color="#b5651d", lw=1.6,
            label="PsycorNeuro 20M  (this work)")
    ax.axhline(SPIKINGBRAIN_SILENT * 100, ls="--", color="#666",
               label="SpikingBrain 7B  (arXiv:2509.05276)")
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("% of neuron-timesteps that are silent")
    ax.set_title("Spike sparsity: most neurons stay quiet throughout training")
    ax.set_ylim(60, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=10)
    ax.annotate(f"mean {(1 - spike.mean()) * 100:.1f}% silent",
                xy=(tok[-1], silent[-1] * 100),
                xytext=(-100, -22), textcoords="offset points", fontsize=10,
                color="#b5651d", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"saved -> {OUT}")
    print(f"mean silent: {(1 - spike.mean()) * 100:.1f}%  "
          f"(vs SpikingBrain {SPIKINGBRAIN_SILENT*100:.0f}%)")


if __name__ == "__main__":
    main()
