"""Analyze frozen-protocol P3-C PMIP4 stress-test results.

This script reads completed P3-C directories only. It validates the frozen
task matrix, summarizes seed variability, bootstraps paired B1/M1-r ensemble
differences over held-out records, and exports analysis tables and figures.
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
OUTPUT = ROOT / "Data" / "model_results" / "v5_pmip4_2d" / "p3c_analysis"
FIGURES = ROOT / "Figures"

SPLITS = ["stress_leave_south", "stress_leave_tropics", "stress_leave_north_excluding_p2dev"]
SPLIT_LABELS = ["Southern high latitudes", "Tropical belt", "Northern high latitudes"]
MODELS = ["B1_pmip4_constd", "M1r_pmip4_relativeD"]
MODEL_LABELS = ["Wind-PINN", "Relative-D PINN"]
MODEL_COLORS = ["#2a6f97", "#c65d3b"]
SEEDS = [42, 520, 2002, 2026, 3407]
EXPECTED_TEST_COUNTS = {
    "stress_leave_south": 36,
    "stress_leave_tropics": 70,
    "stress_leave_north_excluding_p2dev": 175,
}
BOOTSTRAP_SEED = 20260726
N_BOOTSTRAP = 20_000


def load_results() -> tuple[pd.DataFrame, dict[tuple[str, str, int], pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    predictions: dict[tuple[str, str, int], pd.DataFrame] = {}
    for summary_path in sorted(RUNS.glob("p3c_stress_*/**/summary.json")):
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        split, model, seed = result["split"], result["model"], int(result["seed"])
        if model not in MODELS:
            continue
        result["run_id"] = summary_path.parents[2].name
        result["summary_path"] = str(summary_path.relative_to(ROOT)).replace("\\", "/")
        rows.append(result)
        predictions[(split, model, seed)] = pd.read_csv(summary_path.with_name("test_predictions.csv"))
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError(f"No P3-C summaries found below {RUNS}")
    return metrics, predictions


def validate_inputs(metrics: pd.DataFrame, predictions: dict[tuple[str, str, int], pd.DataFrame]) -> None:
    expected = {(split, model, seed) for split in SPLITS for model in MODELS for seed in SEEDS}
    actual = set(zip(metrics.split, metrics.model, metrics.seed.astype(int)))
    if actual != expected:
        raise RuntimeError(f"Frozen P3-C task matrix mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    if not (metrics.epochs_requested.eq(150000).all() and metrics.epochs_completed.eq(150000).all()):
        raise RuntimeError("At least one P3-C model did not complete the 150k-epoch budget")
    if not metrics.prediction_finite_fraction.eq(1.0).all():
        raise RuntimeError("At least one P3-C result has non-finite test predictions")

    for (split, model, seed), frame in predictions.items():
        required = {"record_id", "lon", "lat", "log_dep", "prediction_log_dep", "error_log_dep"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{split}/{model}/seed{seed}: missing columns {sorted(missing)}")
        if len(frame) != EXPECTED_TEST_COUNTS[split] or frame.record_id.duplicated().any():
            raise RuntimeError(f"{split}/{model}/seed{seed}: invalid held-out records")
        if not np.isfinite(frame.prediction_log_dep.to_numpy(dtype=float)).all():
            raise RuntimeError(f"{split}/{model}/seed{seed}: non-finite prediction")
        if "p2_development_record" in frame and frame.p2_development_record.astype(bool).any():
            raise RuntimeError(f"{split}/{model}/seed{seed}: P2 record leaked into test data")


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["rmse", "mae", "r2", "bias", "p95_abs_error"]
    summary = metrics.groupby(["split", "model"], sort=False)[columns].agg(["mean", "std", "min", "max", "count"]).reset_index()
    summary.columns = ["_".join(filter(None, col)).rstrip("_") for col in summary.columns.to_flat_index()]
    return summary


def ensemble_frame(split: str, model: str, predictions: dict[tuple[str, str, int], pd.DataFrame]) -> pd.DataFrame:
    frames = [predictions[(split, model, seed)].set_index("record_id").sort_index() for seed in SEEDS]
    reference = frames[0][["lon", "lat", "log_dep"]].copy()
    for frame in frames[1:]:
        if not reference.index.equals(frame.index) or not np.allclose(reference.log_dep, frame.log_dep):
            raise RuntimeError(f"{split}/{model}: test records differ across seeds")
    matrix = np.column_stack([frame.prediction_log_dep.to_numpy(dtype=float) for frame in frames])
    reference["prediction_mean"] = matrix.mean(axis=1)
    reference["prediction_seed_std"] = matrix.std(axis=1, ddof=1)
    reference["error"] = reference.prediction_mean - reference.log_dep
    reference["abs_error"] = np.abs(reference.error)
    return reference.reset_index()


def bootstrap_b1_vs_m1(predictions: dict[tuple[str, str, int], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries: list[dict[str, object]] = []
    draws: list[pd.DataFrame] = []
    for split in SPLITS:
        b1 = ensemble_frame(split, MODELS[0], predictions).set_index("record_id")
        m1 = ensemble_frame(split, MODELS[1], predictions).set_index("record_id")
        if not b1.index.equals(m1.index) or not np.allclose(b1.log_dep, m1.log_dep):
            raise RuntimeError(f"{split}: B1/M1-r observations cannot be paired")
        observed = b1.log_dep.to_numpy(dtype=float)
        b1_prediction = b1.prediction_mean.to_numpy(dtype=float)
        m1_prediction = m1.prediction_mean.to_numpy(dtype=float)
        indices = rng.integers(0, len(observed), size=(N_BOOTSTRAP, len(observed)))
        b1_error = b1_prediction[indices] - observed[indices]
        m1_error = m1_prediction[indices] - observed[indices]
        series = {
            "rmse": np.sqrt(np.mean(m1_error**2, axis=1)) - np.sqrt(np.mean(b1_error**2, axis=1)),
            "mae": np.mean(np.abs(m1_error), axis=1) - np.mean(np.abs(b1_error), axis=1),
        }
        observed_delta = {
            "rmse": float(np.sqrt(np.mean((m1_prediction - observed) ** 2)) - np.sqrt(np.mean((b1_prediction - observed) ** 2))),
            "mae": float(np.mean(np.abs(m1_prediction - observed)) - np.mean(np.abs(b1_prediction - observed))),
        }
        for metric, values in series.items():
            summaries.append(
                {
                    "split": split,
                    "metric": metric,
                    "n_holdout_records": len(observed),
                    "n_bootstrap": N_BOOTSTRAP,
                    "delta_definition": "M1-r ensemble metric minus B1 ensemble metric; negative favors M1-r",
                    "observed_delta": observed_delta[metric],
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "bootstrap_fraction_delta_lt_zero": float(np.mean(values < 0.0)),
                    "bootstrap_fraction_delta_gt_zero": float(np.mean(values > 0.0)),
                }
            )
            draws.append(pd.DataFrame({"split": split, "metric": metric, "bootstrap_delta": values}))
    return pd.DataFrame(summaries), pd.concat(draws, ignore_index=True)


def table_cell(row: pd.Series, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.3f} +/- {row[f'{metric}_std']:.3f}"


def save_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for split, split_label in zip(SPLITS, SPLIT_LABELS):
        for model, model_label in zip(MODELS, MODEL_LABELS):
            row = summary[(summary.split == split) & (summary.model == model)].iloc[0]
            rows.append({
                "Stress-test region": split_label,
                "Model": model_label,
                "RMSE": table_cell(row, "rmse"),
                "MAE": table_cell(row, "mae"),
                "R2": table_cell(row, "r2"),
                "Bias": table_cell(row, "bias"),
            })
    display = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(10.5, 3.4))
    axis.axis("off")
    table = axis.table(cellText=display.values, colLabels=display.columns, cellLoc="center", loc="center", colWidths=[0.25, 0.18, 0.15, 0.15, 0.14, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.45)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#b0b0b0")
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#e9ecef")
            cell.set_text_props(weight="bold")
        elif row % 2 == 1:
            cell.set_facecolor("#f7f7f7")
    axis.set_title("P3-C frozen stress-test metrics: mean +/- SD across five PINN seeds", pad=12)
    save_figure_all_formats(FIGURES / "V5_P3C_STRESS_METRICS_TABLE", figure)
    plt.close(figure)
    return display


def save_rmse_comparison(summary: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 3.35), constrained_layout=True)
    x = np.arange(len(SPLITS))
    width = 0.32
    for index, (model, label, color) in enumerate(zip(MODELS, MODEL_LABELS, MODEL_COLORS)):
        subset = summary[summary.model == model].set_index("split").loc[SPLITS]
        axis.bar(x + (index - 0.5) * width, subset.rmse_mean, width=width, yerr=subset.rmse_std, capsize=3, color=color, label=label)
    axis.set(xticks=x, xticklabels=["Southern\nhigh latitudes", "Tropical\nbelt", "Northern\nhigh latitudes"], ylabel="Held-out RMSE (log deposition)")
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save_figure_all_formats(FIGURES / "V5_P3C_STRESS_RMSE_COMPARISON", figure)
    plt.close(figure)


def save_bootstrap(bootstrap: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.1), constrained_layout=True)
    for axis, metric, title in zip(axes, ["rmse", "mae"], ["RMSE difference", "MAE difference"]):
        subset = bootstrap[bootstrap.metric == metric].set_index("split").loc[SPLITS]
        values = subset.observed_delta.to_numpy(dtype=float)
        lower = values - subset.ci95_low.to_numpy(dtype=float)
        upper = subset.ci95_high.to_numpy(dtype=float) - values
        colors = ["#2a6f97" if value < 0 else "#c65d3b" for value in values]
        axis.errorbar(range(len(SPLITS)), values, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#343a40", capsize=3, linewidth=0.8)
        axis.scatter(range(len(SPLITS)), values, s=30, c=colors, zorder=3)
        axis.axhline(0.0, color="#343a40", linewidth=0.8)
        axis.set(xticks=range(len(SPLITS)), xticklabels=["South", "Tropics", "North"], ylabel="M1-r ensemble minus B1 ensemble", title=title)
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    figure.suptitle("P3-C paired held-out-record bootstrap, 95% CI; negative favors M1-r", y=1.03)
    save_figure_all_formats(FIGURES / "V5_P3C_M1R_VS_B1_BOOTSTRAP", figure)
    plt.close(figure)


def save_error_maps(predictions: dict[tuple[str, str, int], pd.DataFrame]) -> None:
    ensembles = {(split, model): ensemble_frame(split, model, predictions) for split in SPLITS for model in MODELS}
    vmax = max(float(frame.abs_error.max()) for frame in ensembles.values())
    vmax = max(1.0, np.ceil(vmax * 2.0) / 2.0)
    figure, axes = plt.subplots(3, 2, figsize=(7.0, 6.5), sharex=True, sharey=True, constrained_layout=True)
    scatter = None
    for row, (split, split_label) in enumerate(zip(SPLITS, SPLIT_LABELS)):
        for column, (model, model_label) in enumerate(zip(MODELS, MODEL_LABELS)):
            axis = axes[row, column]
            frame = ensembles[(split, model)]
            scatter = axis.scatter(frame.lon, frame.lat, c=frame.abs_error, cmap="Reds", vmin=0.0, vmax=vmax, s=22, edgecolors="#343a40", linewidths=0.25)
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
    save_figure_all_formats(FIGURES / "V5_P3C_STRESS_ABSOLUTE_ERROR_MAPS", figure)
    plt.close(figure)


def write_manifest(metrics: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    manifest = {
        "purpose": "Frozen P3-C stress-test analysis; no model training or model selection.",
        "p3c_summary_count": int(len(metrics)),
        "pinn_epochs_required": 150000,
        "splits": SPLITS,
        "models": MODELS,
        "bootstrap": {
            "unit": "held-out observation record, paired B1/M1-r ensemble predictions",
            "n_replicates": N_BOOTSTRAP,
            "rng_seed": BOOTSTRAP_SEED,
            "delta_definition": "M1-r ensemble metric minus B1 ensemble metric; negative favors M1-r.",
        },
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
    summary = summarize_metrics(metrics)
    bootstrap, draws = bootstrap_b1_vs_m1(predictions)
    metrics.to_csv(OUTPUT / "p3c_metrics_all_seeds.csv", index=False)
    summary.to_csv(OUTPUT / "p3c_metrics_summary.csv", index=False)
    bootstrap.to_csv(OUTPUT / "p3c_m1r_vs_b1_observation_bootstrap.csv", index=False)
    draws.to_csv(OUTPUT / "p3c_m1r_vs_b1_observation_bootstrap_draws.csv", index=False)
    display = save_table(summary)
    display.to_csv(OUTPUT / "p3c_metrics_display_table.csv", index=False)
    save_rmse_comparison(summary)
    save_bootstrap(bootstrap)
    save_error_maps(predictions)
    write_manifest(metrics, bootstrap)
    print("P3-C analysis completed")
    print(summary.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
