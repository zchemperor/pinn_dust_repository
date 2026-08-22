"""Reproduce the manuscript's outer-confirmation RMSE figure.

This is a plotting-only experiment. It reads the frozen P3-B summary table,
performs no model training, and writes the same figure used in the manuscript
in PNG, EPS, and PDF formats.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
DATA = EXPERIMENT / "data" / "p3b_metrics_summary.csv"
OUTPUT = EXPERIMENT / "output"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_jcp import apply_style, save_figure_all_formats  # noqa: E402


SPLITS = [
    "outer_north_america",
    "outer_east_asia",
    "outer_tropical_atlantic_africa",
]
SPLIT_LABELS = ["North America", "East Asia", "Tropical Atlantic-Africa"]
MODELS = ["B1_pmip4_constd", "M1r_pmip4_relativeD", "B2_nopde_mlp", "B3_gp_rbf"]
MODEL_LABELS = ["Wind-PINN", "Relative-D PINN", "No-PDE MLP", "GP/RBF"]
MODEL_COLORS = ["#2a6f97", "#c65d3b", "#6c757d", "#e0a100"]


def load_summary() -> pd.DataFrame:
    summary = pd.read_csv(DATA)
    required = {"split", "model", "rmse_mean", "rmse_std"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Summary table is missing columns: {sorted(missing)}")
    expected = {(split, model) for split in SPLITS for model in MODELS}
    actual = set(zip(summary["split"], summary["model"]))
    if actual != expected:
        raise ValueError("Summary table does not contain the complete frozen P3-B matrix")
    return summary


def make_figure(summary: pd.DataFrame) -> None:
    figure, axes = plt.subplots(
        1, 3, figsize=(7.0, 3.25), sharey=True, constrained_layout=True
    )
    for axis, split, label in zip(axes, SPLITS, SPLIT_LABELS):
        subset = summary[summary.split == split].set_index("model")
        for index, (model, model_label, color) in enumerate(
            zip(MODELS, MODEL_LABELS, MODEL_COLORS)
        ):
            row = subset.loc[model]
            if model == "B3_gp_rbf":
                axis.plot(
                    index,
                    row["rmse_mean"],
                    marker="D",
                    markersize=5,
                    color=color,
                    linestyle="none",
                    label=model_label,
                )
            else:
                axis.bar(
                    index,
                    row["rmse_mean"],
                    color=color,
                    width=0.68,
                    yerr=row["rmse_std"],
                    capsize=2.5,
                    label=model_label,
                )
        axis.set_title(f"({chr(97 + SPLITS.index(split))}) {label}")
        axis.set_xticks(
            range(len(MODELS)), ["Wind-\nPINN", "Relative-D\nPINN", "No-PDE\nMLP", "GP/RBF"]
        )
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Held-out RMSE (log deposition)")
    save_figure_all_formats(OUTPUT / "outer_confirmation_rmse", figure)
    plt.close(figure)


def main() -> None:
    apply_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    make_figure(load_summary())
    print(f"Created: {OUTPUT / 'outer_confirmation_rmse.png'}")
    print(f"Created: {OUTPUT / 'outer_confirmation_rmse.eps'}")
    print(f"Created: {OUTPUT / 'outer_confirmation_rmse.pdf'}")


if __name__ == "__main__":
    main()
