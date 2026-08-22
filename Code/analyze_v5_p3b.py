"""Analyze frozen-protocol P3-B PMIP4 outer-validation results.

The script reads only completed P3-B result folders. It verifies the frozen
task matrix before writing reproducible tables, observation-level bootstrap
comparisons, and PNG/EPS/PDF figures. It does not train or alter any model.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style_jcp import apply_style, save_figure_all_formats


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "Data" / "model_results" / "v5_pmip4_2d" / "p3_runs"
OUTPUT = ROOT / "Data" / "model_results" / "v5_pmip4_2d" / "p3b_analysis"
FIGURES = ROOT / "Figures"

SPLITS = ["outer_north_america", "outer_east_asia", "outer_tropical_atlantic_africa"]
SPLIT_LABELS = ["North America", "East Asia", "Tropical Atlantic-Africa"]
MODELS = ["B1_pmip4_constd", "M1r_pmip4_relativeD", "B2_nopde_mlp", "B3_gp_rbf"]
MODEL_LABELS = ["Wind-PINN", "Relative-D PINN", "No-PDE MLP", "GP/RBF"]
MODEL_COLORS = ["#2a6f97", "#c65d3b", "#6c757d", "#e0a100"]
PINN_MODELS = MODELS[:3]
SEEDS = [42, 520, 2002, 2026, 3407]
EXPECTED_TEST_COUNTS = {
    "outer_north_america": 64,
    "outer_east_asia": 106,
    "outer_tropical_atlantic_africa": 42,
}
BOOTSTRAP_SEED = 20260725
N_BOOTSTRAP = 20_000


def load_results() -> tuple[pd.DataFrame, dict[tuple[str, str, int], pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    predictions: dict[tuple[str, str, int], pd.DataFrame] = {}
    for summary_path in sorted(RUNS.glob("p3b_outer_*/**/summary.json")):
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        split, model, seed = result["split"], result["model"], int(result["seed"])
        result["run_id"] = summary_path.parents[2].name
        result["summary_path"] = str(summary_path.relative_to(ROOT)).replace("\\", "/")
        rows.append(result)
        prediction_path = summary_path.with_name("test_predictions.csv")
        predictions[(split, model, seed)] = pd.read_csv(prediction_path)

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError(f"No P3-B summaries found below {RUNS}")
    return metrics, predictions


def validate_inputs(metrics: pd.DataFrame, predictions: dict[tuple[str, str, int], pd.DataFrame]) -> None:
    expected = {(split, model, seed) for split in SPLITS for model in PINN_MODELS for seed in SEEDS}
    expected |= {(split, "B3_gp_rbf", 42) for split in SPLITS}
    actual = set(zip(metrics.split, metrics.model, metrics.seed.astype(int)))
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise RuntimeError(f"Frozen P3-B task matrix mismatch; missing={missing}, extra={extra}")

    pinn = metrics[metrics.model.isin(PINN_MODELS)]
    if not (pinn.epochs_requested.eq(150000).all() and pinn.epochs_completed.eq(150000).all()):
        raise RuntimeError("At least one PINN did not complete the frozen 150k-epoch budget")
    if not metrics.prediction_finite_fraction.eq(1.0).all():
        raise RuntimeError("At least one P3-B result has non-finite test predictions")

    for (split, model, seed), frame in predictions.items():
        required = {"record_id", "lon", "lat", "log_dep", "prediction_log_dep", "error_log_dep"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{split}/{model}/seed{seed}: missing prediction columns {sorted(missing)}")
        if len(frame) != EXPECTED_TEST_COUNTS[split] or frame.record_id.duplicated().any():
            raise RuntimeError(f"{split}/{model}/seed{seed}: unexpected held-out record IDs")
        if not np.isfinite(frame.prediction_log_dep.to_numpy(dtype=float)).all():
            raise RuntimeError(f"{split}/{model}/seed{seed}: non-finite held-out prediction")
        if "p2_development_record" in frame and frame.p2_development_record.astype(bool).any():
            raise RuntimeError(f"{split}/{model}/seed{seed}: P2 record leaked into P3 test set")


def metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["rmse", "mae", "r2", "bias", "p95_abs_error"]
    summary = (
        metrics.groupby(["split", "model"], sort=False)[columns]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.columns = ["_".join(filter(None, column)).rstrip("_") for column in summary.columns.to_flat_index()]
    return summary


def ensemble_frame(
    split: str, model: str, predictions: dict[tuple[str, str, int], pd.DataFrame]
) -> pd.DataFrame:
    seeds = SEEDS if model != "B3_gp_rbf" else [42]
    frames = []
    for seed in seeds:
        frame = predictions[(split, model, seed)].set_index("record_id")
        frames.append(frame)
    reference = frames[0][["lon", "lat", "log_dep"]].copy()
    for frame in frames[1:]:
        if not reference.index.equals(frame.index):
            raise RuntimeError(f"{split}/{model}: prediction record ordering differs across seeds")
        if not np.allclose(reference.log_dep, frame.log_dep):
            raise RuntimeError(f"{split}/{model}: observed test data differ across seeds")
    matrix = np.column_stack([frame.prediction_log_dep.to_numpy(dtype=float) for frame in frames])
    reference["prediction_mean"] = matrix.mean(axis=1)
    reference["prediction_seed_std"] = matrix.std(axis=1, ddof=1) if len(frames) > 1 else 0.0
    reference["error"] = reference.prediction_mean - reference.log_dep
    reference["abs_error"] = np.abs(reference.error)
    return reference.reset_index()


def bootstrap_m1r_vs_b1(predictions: dict[tuple[str, str, int], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries: list[dict[str, object]] = []
    draws: list[pd.DataFrame] = []
    for split in SPLITS:
        b1 = ensemble_frame(split, "B1_pmip4_constd", predictions).set_index("record_id")
        m1 = ensemble_frame(split, "M1r_pmip4_relativeD", predictions).set_index("record_id")
        if not b1.index.equals(m1.index) or not np.allclose(b1.log_dep, m1.log_dep):
            raise RuntimeError(f"{split}: B1/M1-r held-out records cannot be paired")

        observed = b1.log_dep.to_numpy(dtype=float)
        b1_pred = b1.prediction_mean.to_numpy(dtype=float)
        m1_pred = m1.prediction_mean.to_numpy(dtype=float)
        indices = rng.integers(0, len(observed), size=(N_BOOTSTRAP, len(observed)))
        b1_error = b1_pred[indices] - observed[indices]
        m1_error = m1_pred[indices] - observed[indices]
        delta_rmse = np.sqrt(np.mean(m1_error**2, axis=1)) - np.sqrt(np.mean(b1_error**2, axis=1))
        delta_mae = np.mean(np.abs(m1_error), axis=1) - np.mean(np.abs(b1_error), axis=1)

        for metric, values in [("rmse", delta_rmse), ("mae", delta_mae)]:
            summaries.append(
                {
                    "split": split,
                    "metric": metric,
                    "n_holdout_records": len(observed),
                    "n_bootstrap": N_BOOTSTRAP,
                    "delta_definition": "M1r ensemble metric minus B1 ensemble metric; negative favors M1-r",
                    "observed_delta": float(np.sqrt(np.mean((m1_pred - observed) ** 2)) - np.sqrt(np.mean((b1_pred - observed) ** 2)))
                    if metric == "rmse"
                    else float(np.mean(np.abs(m1_pred - observed)) - np.mean(np.abs(b1_pred - observed))),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "bootstrap_fraction_delta_lt_zero": float(np.mean(values < 0.0)),
                    "bootstrap_fraction_delta_gt_zero": float(np.mean(values > 0.0)),
                }
            )
            draws.append(pd.DataFrame({"split": split, "metric": metric, "bootstrap_delta": values}))
    return pd.DataFrame(summaries), pd.concat(draws, ignore_index=True)


def save_metric_figure(summary: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 3.25), sharey=True, constrained_layout=True)
    for axis, split, label in zip(axes, SPLITS, SPLIT_LABELS):
        subset = summary[summary.split == split].set_index("model")
        for index, (model, model_label, color) in enumerate(zip(MODELS, MODEL_LABELS, MODEL_COLORS)):
            row = subset.loc[model]
            mean = row["rmse_mean"]
            if model == "B3_gp_rbf":
                axis.plot(index, mean, marker="D", markersize=5, color=color, linestyle="none", label=model_label)
            else:
                axis.bar(index, mean, color=color, width=0.68, yerr=row["rmse_std"], capsize=2.5, label=model_label)
        axis.set_title(f"({chr(97 + SPLITS.index(split))}) {label}")
        axis.set_xticks(range(len(MODELS)), ["Wind-\nPINN", "Relative-D\nPINN", "No-PDE\nMLP", "GP/RBF"])
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Held-out RMSE (log deposition)")
    save_figure_all_formats(FIGURES / "V5_P3B_OUTER_RMSE_COMPARISON", figure)
    plt.close(figure)


def table_cell(row: pd.Series, metric: str) -> str:
    if row["model"] == "B3_gp_rbf":
        return f"{row[f'{metric}_mean']:.3f}"
    return f"{row[f'{metric}_mean']:.3f} +/- {row[f'{metric}_std']:.3f}"


def save_table_figure(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for split, split_label in zip(SPLITS, SPLIT_LABELS):
        for model, model_label in zip(MODELS, MODEL_LABELS):
            record = summary[(summary.split == split) & (summary.model == model)].iloc[0]
            rows.append(
                {
                    "Holdout region": split_label,
                    "Model": model_label,
                    "RMSE": table_cell(record, "rmse"),
                    "MAE": table_cell(record, "mae"),
                    "R2": table_cell(record, "r2"),
                }
            )
    display = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(10.5, 5.1))
    axis.axis("off")
    table = axis.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.24, 0.18, 0.19, 0.19, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#b0b0b0")
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#e9ecef")
            cell.set_text_props(weight="bold")
        elif row % 4 == 1:
            cell.set_facecolor("#f7f7f7")
    axis.set_title("P3-B frozen outer-validation metrics: mean +/- SD across five PINN seeds; B3 is one deterministic fit", pad=12)
    save_figure_all_formats(FIGURES / "V5_P3B_OUTER_METRICS_TABLE", figure)
    plt.close(figure)
    return display


def save_bootstrap_figure(bootstrap: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.1), sharey=False, constrained_layout=True)
    for axis, metric, title in zip(axes, ["rmse", "mae"], ["RMSE difference", "MAE difference"]):
        subset = bootstrap[bootstrap.metric == metric].set_index("split").loc[SPLITS]
        values = subset.observed_delta.to_numpy(dtype=float)
        low = values - subset.ci95_low.to_numpy(dtype=float)
        high = subset.ci95_high.to_numpy(dtype=float) - values
        colors = ["#2a6f97" if value < 0 else "#c65d3b" for value in values]
        axis.errorbar(range(len(SPLITS)), values, yerr=np.vstack([low, high]), fmt="none", ecolor="#343a40", capsize=3, linewidth=0.8)
        axis.scatter(range(len(SPLITS)), values, s=28, c=colors, zorder=3)
        axis.axhline(0.0, color="#343a40", linewidth=0.8)
        axis.set_xticks(range(len(SPLITS)), ["N. America", "E. Asia", "Trop. Atl.-Afr."], rotation=15)
        axis.set_title(title)
        axis.set_ylabel("M1-r ensemble minus B1 ensemble")
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    figure.suptitle("P3-B paired held-out-record bootstrap, 95% CI; negative favors M1-r", y=1.03)
    save_figure_all_formats(FIGURES / "V5_P3B_M1R_VS_B1_BOOTSTRAP", figure)
    plt.close(figure)


def save_error_maps(predictions: dict[tuple[str, str, int], pd.DataFrame]) -> None:
    ensembles = {(split, model): ensemble_frame(split, model, predictions) for split in SPLITS for model in [MODELS[0], MODELS[1], MODELS[3]]}
    vmax = max(float(frame.abs_error.max()) for frame in ensembles.values())
    vmax = max(1.0, np.ceil(vmax * 2.0) / 2.0)
    figure, axes = plt.subplots(3, 3, figsize=(7.0, 6.7), sharex=True, sharey=True, constrained_layout=True)
    scatter = None
    plotted_models = [MODELS[0], MODELS[1], MODELS[3]]
    plotted_labels = [MODEL_LABELS[0], MODEL_LABELS[1], MODEL_LABELS[3]]
    for row, (split, split_label) in enumerate(zip(SPLITS, SPLIT_LABELS)):
        for column, (model, model_label) in enumerate(zip(plotted_models, plotted_labels)):
            axis = axes[row, column]
            frame = ensembles[(split, model)]
            scatter = axis.scatter(
                frame.lon,
                frame.lat,
                c=frame.abs_error,
                cmap="Reds",
                vmin=0.0,
                vmax=vmax,
                s=26,
                marker="o",
                edgecolors="#343a40",
                linewidths=0.25,
            )
            axis.set(xlim=(-180, 180), ylim=(-60, 90), xticks=[-180, -90, 0, 90, 180], yticks=[-60, -30, 0, 30, 60, 90])
            axis.grid(alpha=0.25, linewidth=0.45)
            if row == 0:
                axis.set_title(model_label)
            if column == 0:
                axis.set_ylabel(f"{split_label}\nLatitude (degrees)")
            if row == len(SPLITS) - 1:
                axis.set_xlabel("Longitude (degrees)")
    colorbar = figure.colorbar(scatter, ax=axes, shrink=0.86, pad=0.02)
    colorbar.set_label("Absolute held-out error (log deposition)")
    save_figure_all_formats(FIGURES / "V5_P3B_OUTER_ABSOLUTE_ERROR_MAPS", figure)
    plt.close(figure)


def write_manifest(metrics: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    manifest = {
        "purpose": "Frozen P3-B outer-validation analysis; no model training or model selection.",
        "p3b_summary_count": int(len(metrics)),
        "pinn_runs": int(metrics.model.isin(PINN_MODELS).sum()),
        "gp_runs": int((metrics.model == "B3_gp_rbf").sum()),
        "pinn_epochs_required": 150000,
        "bootstrap": {
            "unit": "held-out observation record, paired B1/M1-r ensemble predictions",
            "n_replicates": N_BOOTSTRAP,
            "rng_seed": BOOTSTRAP_SEED,
            "delta_definition": "M1-r ensemble metric minus B1 ensemble metric; negative favors M1-r.",
        },
        "splits": SPLITS,
        "models": MODELS,
        "bootstrap_rows": int(len(bootstrap)),
    }
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    apply_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics, predictions = load_results()
    validate_inputs(metrics, predictions)
    metrics = metrics.sort_values(["split", "model", "seed"]).reset_index(drop=True)
    summary = metric_summary(metrics)
    bootstrap, draws = bootstrap_m1r_vs_b1(predictions)

    metrics.to_csv(OUTPUT / "p3b_metrics_all_seeds.csv", index=False)
    summary.to_csv(OUTPUT / "p3b_metrics_summary.csv", index=False)
    bootstrap.to_csv(OUTPUT / "p3b_m1r_vs_b1_observation_bootstrap.csv", index=False)
    draws.to_csv(OUTPUT / "p3b_m1r_vs_b1_observation_bootstrap_draws.csv", index=False)
    display = save_table_figure(summary)
    display.to_csv(OUTPUT / "p3b_metrics_display_table.csv", index=False)
    save_metric_figure(summary)
    save_bootstrap_figure(bootstrap)
    save_error_maps(predictions)
    write_manifest(metrics, bootstrap)
    print("P3-B analysis completed")
    print(summary.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
