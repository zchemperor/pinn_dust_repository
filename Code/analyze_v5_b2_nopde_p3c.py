"""Analyze the frozen B2 no-PDE MLP P3-C control against PINN references.

This script consumes completed model outputs only. It does not train, tune, or
select a model. All comparisons use five-seed mean predictions at the identical
held-out records, followed by record-level paired bootstrap intervals.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from style_jcp import apply_style, save_figure_all_formats


ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "Data" / "model_results" / "v5_pmip4_2d"
PINN_RUNS = V5 / "p3_runs"
DIFFUSION_RUNS = V5 / "b2_diffusion_only" / "p3_runs"
CONTROL_PROTOCOL = V5 / "b2_nopde_p3c_protocol.json"
MAP_METRICS = V5 / "pde_map" / "p3c_analysis" / "p3c_pde_map_metrics.csv"
OUTPUT = V5 / "b2_nopde_p3c_analysis"
FIGURES = ROOT / "Figures"
SPLITS = ["stress_leave_south", "stress_leave_tropics", "stress_leave_north_excluding_p2dev"]
SPLIT_LABELS = ["Southern high latitudes", "Tropical belt", "Northern high latitudes"]
SEEDS = [42, 520, 2002, 2026, 3407]
N_TEST = {"stress_leave_south": 36, "stress_leave_tropics": 70, "stress_leave_north_excluding_p2dev": 175}
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 20260729


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    return {
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "bias": float(error.mean()),
        "p95_abs_error": float(np.percentile(np.abs(error), 95)),
    }


def prediction_frame(path: Path, expected_split: str, expected_seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"record_id", "lon", "lat", "log_dep", "prediction_log_dep"}
    if required - set(frame.columns) or len(frame) != N_TEST[expected_split] or frame.record_id.duplicated().any():
        raise RuntimeError(f"{path}: invalid held-out prediction frame")
    if not np.isfinite(frame.prediction_log_dep.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{path}: non-finite prediction")
    if "p2_development_record" in frame and frame.p2_development_record.astype(bool).any():
        raise RuntimeError(f"{path}: P2 development data leaked into test output")
    return frame


def load_nopde() -> tuple[pd.DataFrame, dict[tuple[str, int], pd.DataFrame]]:
    expected_hash = sha256(CONTROL_PROTOCOL)
    rows, predictions = [], {}
    for run in sorted(PINN_RUNS.glob("p3c_stress_*_b2_nopde_seed*")):
        run_protocol = json.loads((run / "experiment_protocol.json").read_text(encoding="utf-8"))
        if run_protocol.get("frozen_control_protocol_sha256") != expected_hash:
            raise RuntimeError(f"{run.name}: missing or mismatched frozen control protocol hash")
        matches = list(run.glob("*/B2_nopde_mlp/summary.json"))
        if len(matches) != 1:
            raise RuntimeError(f"{run.name}: expected one B2 no-PDE summary, found {len(matches)}")
        summary_path = matches[0]
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        split, seed = result["split"], int(result["seed"])
        result["run_id"] = run.name
        result["summary_path"] = str(summary_path.relative_to(ROOT)).replace("\\", "/")
        rows.append(result)
        predictions[(split, seed)] = prediction_frame(summary_path.with_name("test_predictions.csv"), split, seed)
    return pd.DataFrame(rows), predictions


def load_diffusion() -> dict[tuple[str, int], pd.DataFrame]:
    predictions = {}
    for run in sorted(DIFFUSION_RUNS.glob("p3c_stress_*_b2_diffusion_seed*")):
        result = json.loads((run / "b2_metrics.json").read_text(encoding="utf-8"))
        split, seed = result["split"], int(result["seed"])
        path = run / split / "B2_diffusion_only_PINN" / "test_predictions.csv"
        predictions[(split, seed)] = prediction_frame(path, split, seed)
    return predictions


def load_pinn(model_tag: str) -> dict[tuple[str, int], pd.DataFrame]:
    predictions = {}
    for path in PINN_RUNS.glob(f"p3c_stress_*_{model_tag}_seed*/**/test_predictions.csv"):
        result = json.loads(path.with_name("summary.json").read_text(encoding="utf-8"))
        split, seed = result["split"], int(result["seed"])
        predictions[(split, seed)] = prediction_frame(path, split, seed)
    return predictions


def validate_matrix(metrics: pd.DataFrame, *prediction_groups: dict[tuple[str, int], pd.DataFrame]) -> None:
    expected = {(split, seed) for split in SPLITS for seed in SEEDS}
    actual = set(zip(metrics.split, metrics.seed.astype(int)))
    if actual != expected:
        raise RuntimeError(f"B2 no-PDE matrix mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    if not (metrics.epochs_requested.eq(150000).all() and metrics.epochs_completed.eq(150000).all()):
        raise RuntimeError("At least one B2 no-PDE run did not complete 150k epochs")
    if not metrics.prediction_finite_fraction.eq(1.0).all():
        raise RuntimeError("At least one B2 no-PDE run has non-finite predictions")
    for group in prediction_groups:
        if set(group) != expected:
            raise RuntimeError("Reference prediction matrix is incomplete")


def ensemble(split: str, predictions: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    frames = [predictions[(split, seed)].set_index("record_id") for seed in SEEDS]
    base = frames[0][["lon", "lat", "log_dep"]].copy()
    for frame in frames[1:]:
        if not base.index.equals(frame.index) or not np.allclose(base.log_dep, frame.log_dep):
            raise RuntimeError(f"{split}: inconsistent held-out records across seeds")
    matrix = np.column_stack([frame.prediction_log_dep.to_numpy(dtype=float) for frame in frames])
    base["prediction_mean"] = matrix.mean(axis=1)
    base["prediction_seed_sd"] = matrix.std(axis=1, ddof=1)
    return base.reset_index()


def seed_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["rmse", "mae", "r2", "bias", "p95_abs_error", "elapsed_seconds"]
    summary = metrics.groupby("split", sort=False)[columns].agg(["mean", "std", "min", "max", "count"]).reset_index()
    summary.columns = ["_".join(filter(None, item)).rstrip("_") for item in summary.columns.to_flat_index()]
    return summary


def comparison_table(groups: dict[str, dict[tuple[str, int], pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for split, label in zip(SPLITS, SPLIT_LABELS):
        for name, predictions in groups.items():
            frame = ensemble(split, predictions)
            rows.append({"split": split, "split_label": label, "model": name, "n_test": len(frame), **raw_metrics(frame.log_dep.to_numpy(dtype=float), frame.prediction_mean.to_numpy(dtype=float))})
        map_row = pd.read_csv(MAP_METRICS).query("split == @split and model == 'PDE-MAP'")
        if len(map_row) != 1:
            raise RuntimeError(f"{split}: PDE-MAP reference is missing")
        reference = map_row.iloc[0]
        rows.append({"split": split, "split_label": label, "model": "PDE-MAP", "n_test": int(reference.n_test), **{metric: float(reference[metric]) for metric in ["rmse", "mae", "r2", "bias"]}, "p95_abs_error": np.nan})
    return pd.DataFrame(rows)


def paired_bootstrap(
    groups: dict[str, dict[tuple[str, int], pd.DataFrame]], left_name: str, right_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries, draws = [], []
    for split in SPLITS:
        left = ensemble(split, groups[left_name]).set_index("record_id")
        right = ensemble(split, groups[right_name]).set_index("record_id")
        if not left.index.equals(right.index) or not np.allclose(left.log_dep, right.log_dep):
            raise RuntimeError(f"{split}: cannot pair {left_name} and {right_name}")
        observed = left.log_dep.to_numpy(dtype=float)
        left_error = left.prediction_mean.to_numpy(dtype=float) - observed
        right_error = right.prediction_mean.to_numpy(dtype=float) - observed
        indices = rng.integers(0, len(observed), size=(N_BOOTSTRAP, len(observed)))
        for metric, values in [
            ("rmse", np.sqrt(np.mean(left_error[indices] ** 2, axis=1)) - np.sqrt(np.mean(right_error[indices] ** 2, axis=1))),
            ("mae", np.mean(np.abs(left_error[indices]), axis=1) - np.mean(np.abs(right_error[indices]), axis=1)),
        ]:
            observed_delta = float(np.sqrt(np.mean(left_error**2)) - np.sqrt(np.mean(right_error**2))) if metric == "rmse" else float(np.mean(np.abs(left_error)) - np.mean(np.abs(right_error)))
            summaries.append({
                "split": split,
                "metric": metric,
                "n_holdout_records": len(observed),
                "n_bootstrap": N_BOOTSTRAP,
                "delta_definition": f"{left_name} ensemble metric minus {right_name} ensemble metric; negative favors {left_name}",
                "observed_delta": observed_delta,
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "bootstrap_fraction_delta_lt_zero": float(np.mean(values < 0.0)),
            })
            draws.append(pd.DataFrame({"comparison": f"{left_name} minus {right_name}", "split": split, "metric": metric, "bootstrap_delta": values}))
    return pd.DataFrame(summaries), pd.concat(draws, ignore_index=True)


def plot_rmse(comparison: pd.DataFrame) -> None:
    models = ["B1 PMIP4 PINN", "M1-r PINN", "B2 diffusion-only PINN", "B2 no-PDE MLP", "PDE-MAP"]
    colors = ["#2a6f97", "#c65d3b", "#6c757d", "#c29b36", "#5d9c59"]
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True, constrained_layout=True)
    for axis, split, label in zip(axes, SPLITS, SPLIT_LABELS):
        frame = comparison[comparison.split == split].set_index("model").loc[models]
        axis.bar(range(len(models)), frame.rmse, color=colors, width=0.7)
        axis.set(title=label, xticks=range(len(models)), xticklabels=["B1", "M1-r", "B2-d", "MLP", "MAP"])
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Stress-test ensemble RMSE (log deposition)")
    figure.suptitle("Frozen P3-C: no-PDE control and physics-constrained references", y=1.03)
    save_figure_all_formats(FIGURES / "V5_B2_NOPDE_P3C_RMSE_COMPARISON", figure)
    plt.close(figure)


def plot_bootstrap(bootstrap: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    labels = ["South", "Tropics", "North"]
    for axis, metric, title in zip(axes, ["rmse", "mae"], ["RMSE difference", "MAE difference"]):
        frame = bootstrap[bootstrap.metric == metric].set_index("split").loc[SPLITS]
        values = frame.observed_delta.to_numpy(dtype=float)
        lower = values - frame.ci95_low.to_numpy(dtype=float)
        upper = frame.ci95_high.to_numpy(dtype=float) - values
        axis.errorbar(range(3), values, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#343a40", capsize=3, linewidth=0.8)
        axis.scatter(range(3), values, s=30, color="#c29b36", zorder=3)
        axis.axhline(0.0, color="#343a40", linewidth=0.8)
        axis.set(xticks=range(3), xticklabels=labels, title=title, ylabel="no-PDE MLP minus diffusion-only PINN")
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    figure.suptitle("P3-C paired held-out-record bootstrap, 95% CI; negative favors no-PDE", y=1.03)
    save_figure_all_formats(FIGURES / "V5_B2_NOPDE_P3C_VS_DIFFUSION_BOOTSTRAP", figure)
    plt.close(figure)


def plot_seed_rmse(nopde_metrics: pd.DataFrame, diffusion_predictions: dict[tuple[str, int], pd.DataFrame]) -> None:
    rows = []
    for split in SPLITS:
        for seed in SEEDS:
            frame = diffusion_predictions[(split, seed)]
            rows.append({"split": split, "model": "Diffusion-only PINN", "seed": seed, **raw_metrics(frame.log_dep.to_numpy(dtype=float), frame.prediction_log_dep.to_numpy(dtype=float))})
    seed_metrics = pd.concat([nopde_metrics.assign(model="No-PDE MLP"), pd.DataFrame(rows)], ignore_index=True)
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True, constrained_layout=True)
    colors = {"Diffusion-only PINN": "#6c757d", "No-PDE MLP": "#c29b36"}
    for axis, split, label in zip(axes, SPLITS, SPLIT_LABELS):
        frame = seed_metrics[seed_metrics.split == split]
        for xpos, model in enumerate(["Diffusion-only PINN", "No-PDE MLP"]):
            values = frame[frame.model == model].rmse.to_numpy(dtype=float)
            axis.scatter(np.full(len(values), xpos), values, s=28, color=colors[model], zorder=3)
            axis.hlines(values.mean(), xpos - 0.22, xpos + 0.22, color="#343a40", linewidth=1.2)
        axis.set(title=label, xticks=[0, 1], xticklabels=["B2-d", "MLP"])
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Single-seed RMSE (log deposition)")
    figure.suptitle("P3-C seed variability is retained rather than hidden by ensemble means", y=1.03)
    save_figure_all_formats(FIGURES / "V5_B2_NOPDE_P3C_SEED_RMSE", figure)
    plt.close(figure)


def main() -> None:
    apply_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    nopde_metrics, nopde_predictions = load_nopde()
    diffusion_predictions = load_diffusion()
    b1_predictions = load_pinn("b1_constd")
    m1_predictions = load_pinn("m1_relative")
    validate_matrix(nopde_metrics, nopde_predictions, diffusion_predictions, b1_predictions, m1_predictions)
    nopde_metrics = nopde_metrics.sort_values(["split", "seed"]).reset_index(drop=True)
    groups = {
        "B1 PMIP4 PINN": b1_predictions,
        "M1-r PINN": m1_predictions,
        "B2 diffusion-only PINN": diffusion_predictions,
        "B2 no-PDE MLP": nopde_predictions,
    }
    summary = seed_summary(nopde_metrics)
    comparison = comparison_table(groups)
    bootstrap, draws = paired_bootstrap(groups, "B2 no-PDE MLP", "B2 diffusion-only PINN")
    nopde_metrics.to_csv(OUTPUT / "b2_nopde_p3c_metrics_all_seeds.csv", index=False)
    summary.to_csv(OUTPUT / "b2_nopde_p3c_metrics_summary.csv", index=False)
    comparison.to_csv(OUTPUT / "b2_nopde_p3c_ensemble_reference_comparison.csv", index=False)
    bootstrap.to_csv(OUTPUT / "b2_nopde_p3c_vs_diffusion_observation_bootstrap.csv", index=False)
    draws.to_csv(OUTPUT / "b2_nopde_p3c_vs_diffusion_observation_bootstrap_draws.csv", index=False)
    manifest = {
        "purpose": "Frozen P3-C analysis of the B2 no-PDE MLP control against diffusion-only and PMIP4 PINNs; no training or model selection.",
        "formal_nopde_runs": int(len(nopde_metrics)),
        "splits": SPLITS,
        "seeds": SEEDS,
        "epochs_required": 150000,
        "bootstrap": {"n_replicates": N_BOOTSTRAP, "rng_seed": BOOTSTRAP_SEED, "unit": "held-out observation record"},
        "primary_comparison": "B2 no-PDE MLP minus B2 diffusion-only PINN; negative difference favors no-PDE.",
        "interpretation_boundary": "P3-C is a broad-extrapolation stress diagnostic. Relative differences do not establish a globally reliable reconstruction method or physical validity of the source-free steady PDE.",
    }
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_rmse(comparison)
    plot_bootstrap(bootstrap)
    plot_seed_rmse(nopde_metrics, diffusion_predictions)
    print("B2 no-PDE P3-C analysis completed")
    print(comparison.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
