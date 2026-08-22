"""Analyze frozen P4 forcing-sensitivity results against P3-B AWI primary B1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style_jcp import apply_style, save_figure_all_formats


ROOT = Path(__file__).resolve().parents[1]
P3_RUNS = ROOT / "Data" / "model_results" / "v5_pmip4_2d" / "p3_runs"
P4_RUNS = ROOT / "Data" / "model_results" / "v5_pmip4_2d" / "p4_runs"
OUTPUT = ROOT / "Data" / "model_results" / "v5_pmip4_2d" / "p4_analysis"
FIGURES = ROOT / "Figures"

SPLITS = ["outer_north_america", "outer_east_asia", "outer_tropical_atlantic_africa"]
SPLIT_LABELS = ["North America", "East Asia", "Tropical Atlantic-Africa"]
FORCINGS = ["awi_primary", "awi_early", "mpi"]
FORCING_LABELS = ["AWI primary", "AWI early", "MPI"]
FORCING_COLORS = ["#2a6f97", "#4f9c79", "#c65d3b"]
ALTERNATIVES = FORCINGS[1:]
SEEDS = [42, 520, 2002, 2026, 3407]
EXPECTED_TEST_COUNTS = {
    "outer_north_america": 64,
    "outer_east_asia": 106,
    "outer_tropical_atlantic_africa": 42,
}
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 20260727


def forcing_from_run_id(run_id: str) -> str:
    if run_id.startswith("p4_awi_early_"):
        return "awi_early"
    if run_id.startswith("p4_mpi_"):
        return "mpi"
    raise ValueError(f"Unexpected P4 run ID: {run_id}")


def read_summary(summary_path: Path, forcing: str) -> tuple[dict[str, object], pd.DataFrame]:
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    result["forcing"] = forcing
    result["run_id"] = summary_path.parents[2].name
    result["summary_path"] = str(summary_path.relative_to(ROOT)).replace("\\", "/")
    return result, pd.read_csv(summary_path.with_name("test_predictions.csv"))


def load_results() -> tuple[pd.DataFrame, dict[tuple[str, str, int], pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    predictions: dict[tuple[str, str, int], pd.DataFrame] = {}
    for summary_path in sorted(P3_RUNS.glob("p3b_outer_*/**/summary.json")):
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        if result["model"] != "B1_pmip4_constd":
            continue
        row, frame = read_summary(summary_path, "awi_primary")
        key = (row["forcing"], row["split"], int(row["seed"]))
        rows.append(row)
        predictions[key] = frame
    for summary_path in sorted(P4_RUNS.glob("p4_*/**/summary.json")):
        run_id = summary_path.parents[2].name
        if "pathcheck" in run_id:
            continue
        forcing = forcing_from_run_id(run_id)
        row, frame = read_summary(summary_path, forcing)
        key = (row["forcing"], row["split"], int(row["seed"]))
        rows.append(row)
        predictions[key] = frame
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError("No P3-B/P4 results found")
    return metrics, predictions


def validate_inputs(metrics: pd.DataFrame, predictions: dict[tuple[str, str, int], pd.DataFrame]) -> None:
    expected = {(forcing, split, seed) for forcing in FORCINGS for split in SPLITS for seed in SEEDS}
    actual = set(zip(metrics.forcing, metrics.split, metrics.seed.astype(int)))
    if actual != expected:
        raise RuntimeError(f"Frozen P4 comparison matrix mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    if not metrics.model.eq("B1_pmip4_constd").all():
        raise RuntimeError("P4 analysis accepts B1 constD only")
    if not (metrics.epochs_requested.eq(150000).all() and metrics.epochs_completed.eq(150000).all()):
        raise RuntimeError("At least one P3-B/P4 B1 task did not complete 150k epochs")
    if not metrics.prediction_finite_fraction.eq(1.0).all():
        raise RuntimeError("At least one P3-B/P4 B1 prediction is non-finite")
    for (forcing, split, seed), frame in predictions.items():
        required = {"record_id", "lon", "lat", "log_dep", "prediction_log_dep", "error_log_dep"}
        missing = required - set(frame.columns)
        if missing or len(frame) != EXPECTED_TEST_COUNTS[split] or frame.record_id.duplicated().any():
            raise RuntimeError(f"{forcing}/{split}/seed{seed}: invalid held-out prediction file")
        if not np.isfinite(frame.prediction_log_dep.to_numpy(dtype=float)).all():
            raise RuntimeError(f"{forcing}/{split}/seed{seed}: non-finite prediction values")
        if "p2_development_record" in frame and frame.p2_development_record.astype(bool).any():
            raise RuntimeError(f"{forcing}/{split}/seed{seed}: P2 record leaked into held-out data")


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    values = ["rmse", "mae", "r2", "bias", "p95_abs_error"]
    summary = metrics.groupby(["forcing", "split"], sort=False)[values].agg(["mean", "std", "min", "max", "count"]).reset_index()
    summary.columns = ["_".join(filter(None, column)).rstrip("_") for column in summary.columns.to_flat_index()]
    return summary


def ensemble_frame(forcing: str, split: str, predictions: dict[tuple[str, str, int], pd.DataFrame]) -> pd.DataFrame:
    frames = [predictions[(forcing, split, seed)].set_index("record_id").sort_index() for seed in SEEDS]
    reference = frames[0][["lon", "lat", "log_dep"]].copy()
    for frame in frames[1:]:
        if not reference.index.equals(frame.index) or not np.allclose(reference.log_dep, frame.log_dep):
            raise RuntimeError(f"{forcing}/{split}: seed test records differ")
    matrix = np.column_stack([frame.prediction_log_dep.to_numpy(dtype=float) for frame in frames])
    reference["prediction_mean"] = matrix.mean(axis=1)
    reference["prediction_seed_std"] = matrix.std(axis=1, ddof=1)
    reference["error"] = reference.prediction_mean - reference.log_dep
    reference["abs_error"] = np.abs(reference.error)
    return reference.reset_index()


def paired_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[metrics.forcing.eq("awi_primary")].set_index(["split", "seed"])
    rows: list[dict[str, object]] = []
    for forcing in ALTERNATIVES:
        alternative = metrics[metrics.forcing.eq(forcing)].set_index(["split", "seed"])
        for split in SPLITS:
            for seed in SEEDS:
                base, other = primary.loc[(split, seed)], alternative.loc[(split, seed)]
                rows.append({
                    "forcing": forcing, "split": split, "seed": seed,
                    "delta_rmse": float(other.rmse - base.rmse),
                    "delta_mae": float(other.mae - base.mae),
                    "delta_r2": float(other.r2 - base.r2),
                    "delta_bias": float(other.bias - base.bias),
                })
    return pd.DataFrame(rows)


def bootstrap_forcing_differences(predictions: dict[tuple[str, str, int], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries: list[dict[str, object]] = []
    draws: list[pd.DataFrame] = []
    for forcing in ALTERNATIVES:
        for split in SPLITS:
            primary = ensemble_frame("awi_primary", split, predictions).set_index("record_id")
            alternative = ensemble_frame(forcing, split, predictions).set_index("record_id")
            if not primary.index.equals(alternative.index) or not np.allclose(primary.log_dep, alternative.log_dep):
                raise RuntimeError(f"{forcing}/{split}: cannot pair AWI primary and alternative forcing")
            observed = primary.log_dep.to_numpy(dtype=float)
            primary_prediction = primary.prediction_mean.to_numpy(dtype=float)
            alternative_prediction = alternative.prediction_mean.to_numpy(dtype=float)
            indices = rng.integers(0, len(observed), size=(N_BOOTSTRAP, len(observed)))
            primary_error = primary_prediction[indices] - observed[indices]
            alternative_error = alternative_prediction[indices] - observed[indices]
            series = {
                "rmse": np.sqrt(np.mean(alternative_error**2, axis=1)) - np.sqrt(np.mean(primary_error**2, axis=1)),
                "mae": np.mean(np.abs(alternative_error), axis=1) - np.mean(np.abs(primary_error), axis=1),
            }
            for metric, values in series.items():
                observed_delta = (np.sqrt(np.mean((alternative_prediction - observed) ** 2)) - np.sqrt(np.mean((primary_prediction - observed) ** 2))) if metric == "rmse" else (np.mean(np.abs(alternative_prediction - observed)) - np.mean(np.abs(primary_prediction - observed)))
                summaries.append({
                    "forcing": forcing, "split": split, "metric": metric,
                    "n_holdout_records": len(observed), "n_bootstrap": N_BOOTSTRAP,
                    "delta_definition": "alternative-forcing ensemble metric minus AWI-primary ensemble metric; negative favors alternative forcing",
                    "observed_delta": float(observed_delta),
                    "ci95_low": float(np.quantile(values, 0.025)), "ci95_high": float(np.quantile(values, 0.975)),
                    "bootstrap_fraction_delta_lt_zero": float(np.mean(values < 0.0)),
                    "bootstrap_fraction_delta_gt_zero": float(np.mean(values > 0.0)),
                })
                draws.append(pd.DataFrame({"forcing": forcing, "split": split, "metric": metric, "bootstrap_delta": values}))
    return pd.DataFrame(summaries), pd.concat(draws, ignore_index=True)


def table_cell(row: pd.Series, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.3f} +/- {row[f'{metric}_std']:.3f}"


def save_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for split, split_label in zip(SPLITS, SPLIT_LABELS):
        for forcing, forcing_label in zip(FORCINGS, FORCING_LABELS):
            row = summary[(summary.split == split) & (summary.forcing == forcing)].iloc[0]
            rows.append({"Holdout region": split_label, "Forcing": forcing_label, "RMSE": table_cell(row, "rmse"), "MAE": table_cell(row, "mae"), "R2": table_cell(row, "r2"), "Bias": table_cell(row, "bias")})
    display = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(10.5, 4.3))
    axis.axis("off")
    table = axis.table(cellText=display.values, colLabels=display.columns, cellLoc="center", bbox=[0.02, 0.02, 0.96, 0.78], colWidths=[0.25, 0.16, 0.15, 0.15, 0.14, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#b0b0b0")
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#e9ecef")
            cell.set_text_props(weight="bold")
        elif row % 3 == 1:
            cell.set_facecolor("#f7f7f7")
    axis.set_title("P4 forcing sensitivity: B1 held-out metrics, mean +/- SD across five seeds", pad=12)
    save_figure_all_formats(FIGURES / "V5_P4_FORCING_METRICS_TABLE", figure)
    plt.close(figure)
    return display


def save_rmse_comparison(summary: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 3.35), constrained_layout=True)
    x, width = np.arange(len(SPLITS)), 0.23
    for index, (forcing, label, color) in enumerate(zip(FORCINGS, FORCING_LABELS, FORCING_COLORS)):
        subset = summary[summary.forcing.eq(forcing)].set_index("split").loc[SPLITS]
        axis.bar(x + (index - 1) * width, subset.rmse_mean, width=width, yerr=subset.rmse_std, capsize=3, color=color, label=label)
    axis.set(xticks=x, xticklabels=["North\nAmerica", "East\nAsia", "Tropical\nAtlantic-Africa"], ylabel="Held-out RMSE (log deposition)")
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save_figure_all_formats(FIGURES / "V5_P4_FORCING_RMSE_COMPARISON", figure)
    plt.close(figure)


def save_bootstrap(bootstrap: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), constrained_layout=True)
    x = np.arange(len(SPLITS))
    for axis, metric, title in zip(axes, ["rmse", "mae"], ["RMSE difference", "MAE difference"]):
        for index, (forcing, label, color) in enumerate(zip(ALTERNATIVES, FORCING_LABELS[1:], FORCING_COLORS[1:])):
            subset = bootstrap[(bootstrap.metric == metric) & (bootstrap.forcing == forcing)].set_index("split").loc[SPLITS]
            values = subset.observed_delta.to_numpy(dtype=float)
            lower = values - subset.ci95_low.to_numpy(dtype=float)
            upper = subset.ci95_high.to_numpy(dtype=float) - values
            offset = (index - 0.5) * 0.22
            axis.errorbar(x + offset, values, yerr=np.vstack([lower, upper]), fmt="none", ecolor=color, capsize=3, linewidth=0.9)
            axis.scatter(x + offset, values, s=30, c=color, label=label, zorder=3)
        axis.axhline(0.0, color="#343a40", linewidth=0.8)
        axis.set(xticks=x, xticklabels=["North\nAmerica", "East\nAsia", "Tropical\nAtlantic-Africa"], ylabel="Alternative minus AWI-primary\nensemble error (log deposition; negative favors alternative)", title=title)
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[-1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2)
    save_figure_all_formats(FIGURES / "V5_P4_FORCING_BOOTSTRAP", figure)
    plt.close(figure)


def save_error_maps(predictions: dict[tuple[str, str, int], pd.DataFrame]) -> None:
    ensembles = {(forcing, split): ensemble_frame(forcing, split, predictions) for split in SPLITS for forcing in FORCINGS}
    vmax = max(float(frame.abs_error.max()) for frame in ensembles.values())
    vmax = max(1.0, np.ceil(vmax * 2.0) / 2.0)
    figure, axes = plt.subplots(3, 3, figsize=(10.2, 7.8), sharex=True, sharey=True, constrained_layout=True)
    scatter = None
    for row, (split, split_label) in enumerate(zip(SPLITS, SPLIT_LABELS)):
        for column, (forcing, forcing_label) in enumerate(zip(FORCINGS, FORCING_LABELS)):
            axis = axes[row, column]
            frame = ensembles[(forcing, split)]
            scatter = axis.scatter(frame.lon, frame.lat, c=frame.abs_error, cmap="Reds", vmin=0.0, vmax=vmax, s=24, edgecolors="#343a40", linewidths=0.25)
            axis.set(xlim=(-180, 180), ylim=(-60, 90), xticks=[-180, -90, 0, 90, 180], yticks=[-60, -30, 0, 30, 60, 90])
            axis.grid(alpha=0.25, linewidth=0.45)
            if row == 0:
                axis.set_title(forcing_label)
            if column == 0:
                axis.set_ylabel(f"{split_label}\nlatitude (deg)")
            if row == len(SPLITS) - 1:
                axis.set_xlabel("longitude (deg)")
    colorbar = figure.colorbar(scatter, ax=axes, shrink=0.86, pad=0.02)
    colorbar.set_label("Absolute held-out error (log deposition)")
    figure.suptitle("P4 spatial distribution of B1 ensemble-mean held-out errors", y=1.01)
    save_figure_all_formats(FIGURES / "V5_P4_FORCING_ABSOLUTE_ERROR_MAPS", figure)
    plt.close(figure)


def main() -> None:
    apply_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics, predictions = load_results()
    validate_inputs(metrics, predictions)
    metrics = metrics.sort_values(["forcing", "split", "seed"]).reset_index(drop=True)
    summary = summarize(metrics)
    seed_deltas = paired_deltas(metrics)
    bootstrap, draws = bootstrap_forcing_differences(predictions)
    metrics.to_csv(OUTPUT / "p4_metrics_all_seeds.csv", index=False)
    summary.to_csv(OUTPUT / "p4_metrics_summary.csv", index=False)
    seed_deltas.to_csv(OUTPUT / "p4_alternative_vs_primary_seed_deltas.csv", index=False)
    bootstrap.to_csv(OUTPUT / "p4_alternative_vs_primary_observation_bootstrap.csv", index=False)
    draws.to_csv(OUTPUT / "p4_alternative_vs_primary_observation_bootstrap_draws.csv", index=False)
    display = save_table(summary)
    display.to_csv(OUTPUT / "p4_metrics_display_table.csv", index=False)
    save_rmse_comparison(summary)
    save_bootstrap(bootstrap)
    save_error_maps(predictions)
    manifest = {
        "purpose": "Frozen P4 forcing sensitivity against P3-B AWI-primary B1; no model selection.",
        "formal_summary_count": int(len(metrics)), "formal_task_count_required": 45,
        "forcings": FORCINGS, "splits": SPLITS, "seeds": SEEDS,
        "bootstrap": {"unit": "held-out observation record, paired forcing ensemble predictions", "n_replicates": N_BOOTSTRAP, "rng_seed": BOOTSTRAP_SEED},
    }
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("P4 analysis completed")
    print(summary.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
