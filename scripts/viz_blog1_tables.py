"""Render Blog #1's two result tables as PNG images for embedding in Substack
(which doesn't support tables in pasted content)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "assets")

ORANGE = "#b5651d"
LIGHT  = "#fbfaf6"
INK    = "#1a1a1a"
GRAY   = "#999999"


def render_table(title, headers, rows, highlight_rows, out_path,
                 fig_w=7.0, row_h=0.6, head_h=0.7):
    n_rows = len(rows)
    fig_h = head_h + row_h * n_rows + 0.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # title above the table
    ax.text(0, 1.04, title, fontsize=13.5, color=INK, weight="bold",
            transform=ax.transAxes, ha="left", va="bottom")

    col_xs = [0.04, 0.62]   # left-aligned 1st col, value 2nd col
    total_h = head_h + row_h * n_rows
    top_y = 1.00

    # header row
    head_y = top_y - head_h / total_h
    ax.add_patch(plt.Rectangle((0, head_y), 1, head_h / total_h,
                               facecolor=LIGHT, edgecolor="#dadada",
                               linewidth=0.8, transform=ax.transAxes))
    for x, h in zip(col_xs, headers):
        ax.text(x, head_y + (head_h / total_h) / 2, h,
                fontsize=11.5, color=INK, weight="bold",
                transform=ax.transAxes, va="center")

    # body rows
    y = head_y
    for idx, row in enumerate(rows):
        y -= row_h / total_h
        face = "#fdf3e7" if idx in highlight_rows else "white"
        ax.add_patch(plt.Rectangle((0, y), 1, row_h / total_h,
                                   facecolor=face, edgecolor="#dadada",
                                   linewidth=0.5, transform=ax.transAxes))
        label, value = row
        ax.text(col_xs[0], y + (row_h / total_h) / 2, label,
                fontsize=11.5, color=INK, transform=ax.transAxes, va="center")
        ax.text(col_xs[1], y + (row_h / total_h) / 2, value,
                fontsize=12.0, color=ORANGE if idx in highlight_rows else INK,
                weight="bold" if idx in highlight_rows else "normal",
                transform=ax.transAxes, va="center")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Result 1: controlled induction task
    render_table(
        title="Result 1 — controlled induction task",
        headers=["Setting", "Recall accuracy on the second copy"],
        rows=[
            ("With synaptic memory M", "96%"),
            ("Synaptic memory zeroed", "5%"),
            ("Random chance",          "5%"),
        ],
        highlight_rows=[0],   # bold the winning row
        out_path=os.path.join(OUT_DIR, "blog1_result1_table.png"),
    )

    # Result 2: 700M Llama-3 tokens
    render_table(
        title="Result 2 — 700M Llama-3 tokens, 20M-param model",
        headers=["Setting", "Validation perplexity (lower = better)"],
        rows=[
            ("With synaptic memory M",                "89.10"),
            ("Synaptic memory zeroed",                "176.34"),
            ("Δ when memory is removed",              "+87.24 ppl  (98% jump)"),
        ],
        highlight_rows=[2],   # the Δ is the punchline
        out_path=os.path.join(OUT_DIR, "blog1_result2_table.png"),
    )


if __name__ == "__main__":
    main()
