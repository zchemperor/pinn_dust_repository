"""Draw the manuscript workflow for the frozen PMIP4-informed study."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from style_jcp import apply_style, save_figure_all_formats


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "Figures"


def box(axis, xy, width, height, title, detail, color):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.0, edgecolor="#343a40", facecolor=color,
    )
    axis.add_patch(patch)
    x, y = xy
    axis.text(x + width / 2, y + height * 0.67, title, ha="center", va="center", fontsize=9, fontweight="bold")
    axis.text(x + width / 2, y + height * 0.34, detail, ha="center", va="center", fontsize=7.4, linespacing=1.15)


def arrow(axis, start, end):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.0, color="#343a40"))


def main() -> None:
    apply_style()
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.set(xlim=(0, 7.0), ylim=(0, 4.5))
    axis.axis("off")

    box(axis, (0.25, 3.0), 2.0, 0.9, "Sparse LGM data", "DIRTMAP log deposition\nn = 317 records", "#dcecf4")
    box(axis, (0.25, 1.05), 2.0, 0.9, "PMIP4 forcing", "Annual LGM 850 hPa U,V\nAWI primary; alternatives", "#e4f0df")
    box(axis, (2.5, 2.0), 2.0, 1.35, "Wind-PINN", "Spherical advection-diffusion\nregularization; constant D", "#e9e1f2")
    box(axis, (4.75, 3.0), 2.0, 0.9, "Frozen evaluation", "Regional confirmation\nExtensive-extrapolation stress;\nforcing sensitivity", "#f6efc8")
    box(axis, (4.75, 1.05), 2.0, 0.9, "Controls", "No-PDE MLP; GP/RBF; PDE-MAP\nRelative-D coefficient ablation", "#f8e7d5")

    arrow(axis, (2.25, 3.45), (2.5, 2.95))
    arrow(axis, (2.25, 1.5), (2.5, 2.4))
    arrow(axis, (4.5, 2.9), (4.75, 3.45))
    arrow(axis, (5.75, 1.95), (5.75, 3.0))

    axis.text(3.5, 0.42, "Regional comparisons use matched baselines; forcing sensitivity and extrapolation limits are reported explicitly.", ha="center", va="center", fontsize=7.6, color="#343a40")
    save_figure_all_formats(FIGURES / "workflow_pmip4", figure)
    plt.close(figure)


if __name__ == "__main__":
    main()
