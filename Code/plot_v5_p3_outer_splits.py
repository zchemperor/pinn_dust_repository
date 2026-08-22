"""Plot frozen v5 P3 outer confirmation splits in PNG and EPS formats."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from style_jcp import apply_style, save_figure_all_formats


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "Data" / "model_results" / "v5_pmip4_2d" / "p3_protocol"
FIGURES = ROOT / "Figures"
OUTER = ["outer_north_america", "outer_east_asia", "outer_tropical_atlantic_africa"]
TITLES = ["North America", "East Asia", "Tropical Atlantic-Africa"]


def main() -> None:
    apply_style()
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharex=True, sharey=True, constrained_layout=True)
    for axis, name, title in zip(axes, OUTER, TITLES):
        train = pd.read_csv(SPLITS / name / "train_records.csv")
        test = pd.read_csv(SPLITS / name / "test_records.csv")
        p2_dev = train[train["p2_development_record"]]
        remaining = train[~train["p2_development_record"]]
        axis.scatter(remaining["lon"], remaining["lat"], s=15, color="#6c757d", label=f"train (n={len(train)})")
        axis.scatter(test["lon"], test["lat"], s=24, color="#c65d3b", label=f"outer test (n={len(test)})")
        axis.scatter(p2_dev["lon"], p2_dev["lat"], s=24, color="#e0a100", marker="s", label=f"P2 development in train (n={len(p2_dev)})")
        axis.set(title=f"({chr(97 + OUTER.index(name))}) {title}", xlim=(-180, 180), ylim=(-85, 85), xlabel="Longitude (degrees)")
        axis.set_xticks(range(-180, 181, 90))
        axis.set_yticks(range(-60, 61, 30))
    axes[0].set_ylabel("Latitude (degrees)")

    legend_axis = axes[-1]
    legend_x = -164
    legend_text_x = -151
    legend_y = [-59, -70, -81]
    legend_items = [
        ("#6c757d", "o", 15, f"Train (n={len(train)})"),
        ("#c65d3b", "o", 24, f"Outer test (n={len(test)})"),
        ("#e0a100", "s", 24, f"P2 development (n={len(p2_dev)})"),
    ]
    for y, (color, marker, size, label) in zip(legend_y, legend_items):
        legend_axis.scatter([legend_x], [y], s=size, color=color, marker=marker, clip_on=False, zorder=4)
        legend_axis.text(legend_text_x, y, label, va="center", fontsize=7, color="black", zorder=4)
    save_figure_all_formats(FIGURES / "V5_P3_OUTER_SPLITS", fig=figure, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()
