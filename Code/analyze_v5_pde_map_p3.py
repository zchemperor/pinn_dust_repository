"""Compare frozen P3-B PMIP4 PINN, static PDE-MAP, and GP/RBF predictions.

The static MAP baseline is deterministic.  PINN comparisons therefore use the
mean prediction across the five frozen B1 seeds, while the original PINN
mean-plus-SD summary is retained for reporting initialization variation.
Bootstrap intervals resample held-out records and are descriptive conditional
comparisons, not calibrated predictive uncertainty.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from style_jcp import apply_style, save_figure_all_formats


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "Data" / "model_results" / "v5_pmip4_2d" / "p3_runs"
MAP_RUNS = ROOT / "Data" / "model_results" / "v5_pmip4_2d" / "pde_map"
OLD_ANALYSIS = ROOT / "Data" / "model_results" / "v5_pmip4_2d" / "p3b_analysis"
OUTPUT = MAP_RUNS / "p3_analysis"
FIGURES = ROOT / "Figures"

SPLITS = ["outer_north_america", "outer_east_asia", "outer_tropical_atlantic_africa"]
SPLIT_LABELS = {
    "outer_north_america": "North America",
    "outer_east_asia": "East Asia",
    "outer_tropical_atlantic_africa": "Tropical Atlantic-Africa",
}
SEEDS = [42, 520, 2002, 2026, 3407]
RNG = np.random.default_rng(20260727)
BOOTSTRAP_DRAWS = 20_000


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    return {
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "bias": float(error.mean()),
        "p95_abs_error": float(np.percentile(np.abs(error), 95)),
    }


def prediction_path(split: str, model: str, seed: int | None = None) -> Path:
    if model == "pinn":
        if seed is None:
            raise ValueError("PINN predictions require a seed")
        return RUNS / f"p3b_{split}_b1_constd_seed{seed}" / split / "B1_pmip4_constd" / "test_predictions.csv"
    if model == "gp":
        return RUNS / f"p3b_{split}_b3_gp_seed42" / split / "B3_gp_rbf" / "test_predictions.csv"
    if model == "map":
        return MAP_RUNS / f"p3_formal_{split}" / "test_predictions.csv"
    raise ValueError(model)


def load_split_predictions(split: str) -> pd.DataFrame:
    seed_frames = [pd.read_csv(prediction_path(split, "pinn", seed)).set_index("record_id").sort_index() for seed in SEEDS]
    reference = seed_frames[0][["lon", "lat", "log_dep"]].copy()
    for frame in seed_frames[1:]:
        if not reference.index.equals(frame.index) or not np.allclose(reference.log_dep, frame.log_dep):
            raise RuntimeError(f"{split}: PINN seeds have inconsistent held-out records")
    map_frame = pd.read_csv(prediction_path(split, "map")).set_index("record_id").sort_index()
    gp_frame = pd.read_csv(prediction_path(split, "gp")).set_index("record_id").sort_index()
    for label, frame in [("PDE-MAP", map_frame), ("GP/RBF", gp_frame)]:
        if not reference.index.equals(frame.index) or not np.allclose(reference.log_dep, frame.log_dep):
            raise RuntimeError(f"{split}: {label} held-out records do not match the frozen PINN split")
    predictions = np.column_stack([frame.prediction_log_dep.to_numpy(float) for frame in seed_frames])
    output = reference.copy()
    output["pinn_prediction_mean"] = predictions.mean(axis=1)
    output["pinn_prediction_seed_sd"] = predictions.std(axis=1, ddof=1)
    output["map_prediction"] = map_frame.prediction_log_dep.to_numpy(float)
    output["gp_prediction"] = gp_frame.prediction_log_dep.to_numpy(float)
    output["split"] = split
    for column in ["pinn_prediction_mean", "map_prediction", "gp_prediction"]:
        output[f"{column}_error"] = output[column] - output.log_dep
    return output.reset_index()


def old_pinn_summary(split: str) -> pd.Series:
    summary = pd.read_csv(OLD_ANALYSIS / "p3b_metrics_summary.csv")
    row = summary[(summary.split == split) & (summary.model == "B1_pmip4_constd")]
    if len(row) != 1:
        raise RuntimeError(f"{split}: missing frozen B1 PINN summary")
    return row.iloc[0]


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    methods = {
        "PDE-MAP": "map_prediction",
        "B1 PINN ensemble": "pinn_prediction_mean",
        "B3 GP/RBF": "gp_prediction",
    }
    for split in SPLITS:
        frame = records[records.split == split]
        for label, column in methods.items():
            row = {"split": split, "split_label": SPLIT_LABELS[split], "model": label, "n_test": len(frame), **metrics(frame.log_dep.to_numpy(float), frame[column].to_numpy(float))}
            if label == "B1 PINN ensemble":
                frozen = old_pinn_summary(split)
                row["pinn_seed_metric_rmse_mean"] = frozen.rmse_mean
                row["pinn_seed_metric_rmse_sd"] = frozen.rmse_std
                row["pinn_seed_metric_mae_mean"] = frozen.mae_mean
                row["pinn_seed_metric_mae_sd"] = frozen.mae_std
            rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, draws = [], []
    for split in SPLITS:
        frame = records[records.split == split].reset_index(drop=True)
        observed = frame.log_dep.to_numpy(float)
        map_prediction = frame.map_prediction.to_numpy(float)
        pinn_prediction = frame.pinn_prediction_mean.to_numpy(float)
        n = len(frame)
        indices = RNG.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
        map_error = map_prediction[indices] - observed[indices]
        pinn_error = pinn_prediction[indices] - observed[indices]
        rmse_difference = np.sqrt(np.mean(map_error**2, axis=1)) - np.sqrt(np.mean(pinn_error**2, axis=1))
        mae_difference = np.mean(np.abs(map_error), axis=1) - np.mean(np.abs(pinn_error), axis=1)
        for metric_name, values in [("rmse", rmse_difference), ("mae", mae_difference)]:
            rows.append({
                "split": split,
                "split_label": SPLIT_LABELS[split],
                "comparison": "PDE-MAP minus B1 PINN ensemble",
                "metric": metric_name,
                "observed_difference": float(values.mean()),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "negative_favors": "PDE-MAP",
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "n_test": n,
            })
            draws.append(pd.DataFrame({"split": split, "metric": metric_name, "difference_map_minus_pinn": values}))
    return pd.DataFrame(rows), pd.concat(draws, ignore_index=True)


def save_comparison_figure(summary: pd.DataFrame) -> None:
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    method_order = ["PDE-MAP", "B1 PINN ensemble", "B3 GP/RBF"]
    colors = {"PDE-MAP": "#6a994e", "B1 PINN ensemble": "#2a6f97", "B3 GP/RBF": "#c65d3b"}
    positions = np.arange(len(SPLITS))
    width = 0.24
    for axis, metric in zip(axes, ["rmse", "mae"]):
        for index, method in enumerate(method_order):
            subset = summary[summary.model == method].set_index("split").loc[SPLITS]
            offset = (index - 1) * width
            error = subset.get(f"pinn_seed_metric_{metric}_sd") if method == "B1 PINN ensemble" else None
            axis.bar(positions + offset, subset[metric], width=width, color=colors[method], label=method, yerr=error, capsize=2.5)
        axis.set(xticks=positions, xticklabels=[SPLIT_LABELS[item] for item in SPLITS], ylabel=f"Held-out {metric.upper()} (log deposition)", title=f"P3-B {metric.upper()}: lower is better")
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
        axis.tick_params(axis="x", rotation=18)
    axes[0].legend(frameon=False, loc="upper left")
    figure.suptitle("Frozen P3-B comparison: static PDE-MAP, B1 PINN, and GP/RBF", y=1.03)
    save_figure_all_formats(FIGURES / "V5_P3B_PDE_MAP_COMPARISON", figure)
    plt.close(figure)


def save_table_figure(summary: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    display = summary.pivot(index="split_label", columns="model", values="rmse").reindex(columns=["PDE-MAP", "B1 PINN ensemble", "B3 GP/RBF"]).reset_index()
    display.columns = ["Holdout", "PDE-MAP RMSE", "PINN RMSE", "GP/RBF RMSE"]
    map_delta = bootstrap[bootstrap.metric == "rmse"].set_index("split_label")
    display["MAP-PINN RMSE [95% CI]"] = display.Holdout.map(
        lambda label: f"{map_delta.loc[label, 'observed_difference']:.3f} [{map_delta.loc[label, 'ci95_low']:.3f}, {map_delta.loc[label, 'ci95_high']:.3f}]"
    )
    for column in ["PDE-MAP RMSE", "PINN RMSE", "GP/RBF RMSE"]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    apply_style()
    figure, axis = plt.subplots(figsize=(11.0, 2.5))
    axis.axis("off")
    table = axis.table(cellText=display.values, colLabels=display.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.55)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#b0b0b0")
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#e9ecef")
            cell.set_text_props(weight="bold")
        elif row % 2 == 1:
            cell.set_facecolor("#f7f7f7")
    axis.set_title("P3-B static PDE-MAP comparison; negative MAP-PINN difference favors PDE-MAP", pad=10)
    save_figure_all_formats(FIGURES / "V5_P3B_PDE_MAP_TABLE", figure)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = pd.concat([load_split_predictions(split) for split in SPLITS], ignore_index=True)
    if not np.isfinite(records.select_dtypes(include=np.number).to_numpy()).all():
        raise RuntimeError("Non-finite value found in the P3-B comparison")
    summary = summarize(records)
    bootstrap, draws = paired_bootstrap(records)
    records.to_csv(OUTPUT / "p3_pde_map_record_comparison.csv", index=False)
    summary.to_csv(OUTPUT / "p3_pde_map_metrics.csv", index=False)
    bootstrap.to_csv(OUTPUT / "p3_pde_map_vs_pinn_bootstrap.csv", index=False)
    draws.to_csv(OUTPUT / "p3_pde_map_vs_pinn_bootstrap_draws.csv", index=False)
    manifest = {
        "purpose": "Frozen P3-B comparison of static PDE-MAP with the five-seed B1 PINN ensemble and existing GP/RBF.",
        "splits": SPLITS,
        "pinn_seeds": SEEDS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_interpretation": "Conditional held-out-record resampling; not calibrated predictive uncertainty.",
        "record_rows": int(len(records)),
        "metric_rows": int(len(summary)),
    }
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save_comparison_figure(summary)
    save_table_figure(summary, bootstrap)
    print(summary.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
