from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from style_jcp import apply_style, save_figure_all_formats


ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "Data" / "model_results" / "v5_pmip4_2d"
B2_RUNS = V5 / "b2_diffusion_only" / "p3_runs"
PINN_RUNS = V5 / "p3_runs"
MAP_METRICS = V5 / "pde_map" / "p3c_analysis" / "p3c_pde_map_metrics.csv"
OUTPUT = V5 / "b2_diffusion_only" / "p3c_analysis"
FIGURES = ROOT / "Figures"
SPLITS = ["stress_leave_south", "stress_leave_tropics", "stress_leave_north_excluding_p2dev"]
SPLIT_LABELS = ["Southern high latitudes", "Tropical belt", "Northern high latitudes"]
SEEDS = [42, 520, 2002, 2026, 3407]
N_TEST = {"stress_leave_south": 36, "stress_leave_tropics": 70, "stress_leave_north_excluding_p2dev": 175}
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 20260728


def raw_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    return {
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "bias": float(error.mean()),
        "p95_abs_error": float(np.percentile(np.abs(error), 95)),
    }


def load_b2() -> tuple[pd.DataFrame, dict[tuple[str, int], pd.DataFrame]]:
    rows, predictions = [], {}
    for run in sorted(B2_RUNS.glob("p3c_stress_*_b2_diffusion_seed*")):
        metrics_path = run / "b2_metrics.json"
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        split, seed = result["split"], int(result["seed"])
        prediction_path = run / split / "B2_diffusion_only_PINN" / "test_predictions.csv"
        result["run_id"] = run.name
        result["metrics_path"] = str(metrics_path.relative_to(ROOT)).replace("\\", "/")
        rows.append(result)
        predictions[(split, seed)] = pd.read_csv(prediction_path)
    return pd.DataFrame(rows), predictions


def load_pinn(model_tag: str) -> dict[tuple[str, int], pd.DataFrame]:
    outputs = {}
    for path in PINN_RUNS.glob(f"p3c_stress_*_{model_tag}_seed*/**/test_predictions.csv"):
        result = json.loads(path.with_name("summary.json").read_text(encoding="utf-8"))
        outputs[(result["split"], int(result["seed"]))] = pd.read_csv(path)
    return outputs


def validate(
    b2_metrics: pd.DataFrame,
    b2_predictions: dict[tuple[str, int], pd.DataFrame],
    b1_predictions: dict[tuple[str, int], pd.DataFrame],
    m1_predictions: dict[tuple[str, int], pd.DataFrame],
) -> None:
    expected = {(split, seed) for split in SPLITS for seed in SEEDS}
    actual = set(zip(b2_metrics.split, b2_metrics.seed.astype(int)))
    if actual != expected:
        raise RuntimeError(f"B2 P3-C matrix mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    if not (b2_metrics.epochs_requested.eq(150000).all() and b2_metrics.epochs_completed.eq(150000).all()):
        raise RuntimeError("At least one B2 task did not complete 150k epochs")
    if not b2_metrics.prediction_finite_fraction.eq(1.0).all():
        raise RuntimeError("At least one B2 task has non-finite predictions")
    if set(b2_predictions) != expected or set(b1_predictions) != expected or set(m1_predictions) != expected:
        raise RuntimeError("B1/M1-r/B2 prediction matrix is incomplete")
    required = {"record_id", "lon", "lat", "log_dep", "prediction_log_dep"}
    for key, frame in {**b2_predictions, **b1_predictions, **m1_predictions}.items():
        if required - set(frame.columns) or len(frame) != N_TEST[key[0]] or frame.record_id.duplicated().any():
            raise RuntimeError(f"{key}: invalid held-out prediction frame")
        if not np.isfinite(frame.prediction_log_dep.to_numpy(dtype=float)).all():
            raise RuntimeError(f"{key}: non-finite held-out prediction")
        if "p2_development_record" in frame and frame.p2_development_record.astype(bool).any():
            raise RuntimeError(f"{key}: P2 development record leaked into test data")


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


def b2_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["rmse", "mae", "r2", "bias", "p95_abs_error", "d_nd_constant", "elapsed_seconds"]
    frame = metrics.groupby("split", sort=False)[columns].agg(["mean", "std", "min", "max", "count"]).reset_index()
    frame.columns = ["_".join(filter(None, item)).rstrip("_") for item in frame.columns.to_flat_index()]
    return frame


def comparison_table(
    b1_predictions: dict[tuple[str, int], pd.DataFrame],
    m1_predictions: dict[tuple[str, int], pd.DataFrame],
    b2_predictions: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    map_frame = pd.read_csv(MAP_METRICS)
    rows = []
    for split, label in zip(SPLITS, SPLIT_LABELS):
        for name, predictions in [
            ("B1 PMIP4 PINN", b1_predictions),
            ("M1-r PINN", m1_predictions),
            ("B2 zero-wind PINN", b2_predictions),
        ]:
            frame = ensemble(split, predictions)
            rows.append({"split": split, "split_label": label, "model": name, "n_test": len(frame), **raw_metrics(frame.log_dep.to_numpy(dtype=float), frame.prediction_mean.to_numpy(dtype=float))})
        matches = map_frame[(map_frame["split"] == split) & (map_frame["model"] == "PDE-MAP")]
        if len(matches) != 1:
            raise RuntimeError(f"{split}: missing PDE-MAP reference")
        reference = matches.iloc[0]
        rows.append({"split": split, "split_label": label, "model": "PDE-MAP", "n_test": int(reference.n_test), **{name: float(reference[name]) for name in ["rmse", "mae", "r2", "bias"]}, "p95_abs_error": np.nan})
    return pd.DataFrame(rows)


def bootstrap_b2_vs_b1(
    b1_predictions: dict[tuple[str, int], pd.DataFrame], b2_predictions: dict[tuple[str, int], pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary, draws = [], []
    for split in SPLITS:
        b1, b2 = ensemble(split, b1_predictions).set_index("record_id"), ensemble(split, b2_predictions).set_index("record_id")
        if not b1.index.equals(b2.index) or not np.allclose(b1.log_dep, b2.log_dep):
            raise RuntimeError(f"{split}: B1/B2 predictions cannot be paired")
        observed = b1.log_dep.to_numpy(dtype=float)
        b1_error = b1.prediction_mean.to_numpy(dtype=float) - observed
        b2_error = b2.prediction_mean.to_numpy(dtype=float) - observed
        indices = rng.integers(0, len(observed), size=(N_BOOTSTRAP, len(observed)))
        for metric, values in [
            ("rmse", np.sqrt(np.mean(b2_error[indices] ** 2, axis=1)) - np.sqrt(np.mean(b1_error[indices] ** 2, axis=1))),
            ("mae", np.mean(np.abs(b2_error[indices]), axis=1) - np.mean(np.abs(b1_error[indices]), axis=1)),
        ]:
            observed_delta = float(np.sqrt(np.mean(b2_error**2)) - np.sqrt(np.mean(b1_error**2))) if metric == "rmse" else float(np.mean(np.abs(b2_error)) - np.mean(np.abs(b1_error)))
            summary.append({
                "split": split,
                "metric": metric,
                "n_holdout_records": len(observed),
                "n_bootstrap": N_BOOTSTRAP,
                "delta_definition": "B2 zero-wind ensemble metric minus B1 PMIP4-wind ensemble metric; positive favors B1",
                "observed_delta": observed_delta,
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "bootstrap_fraction_delta_gt_zero": float(np.mean(values > 0.0)),
            })
            draws.append(pd.DataFrame({"split": split, "metric": metric, "bootstrap_delta": values}))
    return pd.DataFrame(summary), pd.concat(draws, ignore_index=True)


def plot_metrics(comparison: pd.DataFrame) -> None:
    models = ["B1 PMIP4 PINN", "M1-r PINN", "B2 zero-wind PINN", "PDE-MAP"]
    colors = ["#2a6f97", "#c65d3b", "#6c757d", "#5d9c59"]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True, constrained_layout=True)
    for axis, split, label in zip(axes, SPLITS, SPLIT_LABELS):
        frame = comparison[comparison.split == split].set_index("model").loc[models]
        axis.bar(range(len(models)), frame.rmse, color=colors, width=0.68)
        axis.set(title=label, xticks=range(len(models)), xticklabels=["B1", "M1-r", "B2-z", "MAP"])
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Stress-test ensemble RMSE (log deposition)")
    figure.suptitle("Frozen P3-C stress test: zero-wind B2 and reference models", y=1.03)
    save_figure_all_formats(FIGURES / "V5_B2_P3C_RMSE_COMPARISON", figure)
    plt.close(figure)


def plot_bootstrap(bootstrap: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    labels = ["South", "Tropics", "North"]
    for axis, metric, title in zip(axes, ["rmse", "mae"], ["RMSE difference", "MAE difference"]):
        frame = bootstrap[bootstrap.metric == metric].set_index("split").loc[SPLITS]
        values = frame.observed_delta.to_numpy(dtype=float)
        low, high = values - frame.ci95_low.to_numpy(dtype=float), frame.ci95_high.to_numpy(dtype=float) - values
        axis.errorbar(range(3), values, yerr=np.vstack([low, high]), fmt="none", ecolor="#343a40", capsize=3, linewidth=0.8)
        axis.scatter(range(3), values, s=30, color="#c65d3b", zorder=3)
        axis.axhline(0.0, color="#343a40", linewidth=0.8)
        axis.set(xticks=range(3), xticklabels=labels, title=title, ylabel="B2 zero-wind minus B1 PMIP4-wind")
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    figure.suptitle("P3-C paired held-out-record bootstrap, 95% CI; positive favors B1", y=1.03)
    save_figure_all_formats(FIGURES / "V5_B2_P3C_VS_B1_BOOTSTRAP", figure)
    plt.close(figure)


def plot_diffusivity(summary: pd.DataFrame) -> None:
    frame = summary.set_index("split").loc[SPLITS]
    scale = 1e6
    centers = (frame.d_nd_constant_mean.to_numpy(dtype=float) - 0.02) * scale
    low = centers - (frame.d_nd_constant_min.to_numpy(dtype=float) - 0.02) * scale
    high = (frame.d_nd_constant_max.to_numpy(dtype=float) - 0.02) * scale - centers
    figure, axis = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    axis.errorbar(range(3), centers, yerr=np.vstack([low, high]), fmt="o", color="#c65d3b", capsize=4)
    axis.axhline(0.0, color="#343a40", linestyle="--", linewidth=0.9, label="parameter lower bound")
    axis.set(xticks=range(3), xticklabels=SPLIT_LABELS, ylabel=r"B2 learned $D_{nd} - 0.02$ ($10^{-6}$)", title="Zero-wind B2 diffusivity remains at the lower bound")
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False)
    save_figure_all_formats(FIGURES / "V5_B2_P3C_DIFFUSIVITY_BOUND", figure)
    plt.close(figure)


def main() -> None:
    apply_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    b2_metrics, b2_predictions = load_b2()
    b1_predictions = load_pinn("b1_constd")
    m1_predictions = load_pinn("m1_relative")
    validate(b2_metrics, b2_predictions, b1_predictions, m1_predictions)
    b2_metrics = b2_metrics.sort_values(["split", "seed"]).reset_index(drop=True)
    summary = b2_summary(b2_metrics)
    comparison = comparison_table(b1_predictions, m1_predictions, b2_predictions)
    bootstrap, draws = bootstrap_b2_vs_b1(b1_predictions, b2_predictions)
    b2_metrics.to_csv(OUTPUT / "b2_p3c_metrics_all_seeds.csv", index=False)
    summary.to_csv(OUTPUT / "b2_p3c_metrics_summary.csv", index=False)
    comparison.to_csv(OUTPUT / "b2_p3c_ensemble_reference_comparison.csv", index=False)
    bootstrap.to_csv(OUTPUT / "b2_p3c_vs_b1_observation_bootstrap.csv", index=False)
    draws.to_csv(OUTPUT / "b2_p3c_vs_b1_observation_bootstrap_draws.csv", index=False)
    manifest = {
        "purpose": "Frozen P3-C stress-test analysis of B2 diffusion-only PINN; no training or model selection.",
        "formal_b2_runs": int(len(b2_metrics)),
        "splits": SPLITS,
        "seeds": SEEDS,
        "epochs_required": 150000,
        "bootstrap": {"n_replicates": N_BOOTSTRAP, "rng_seed": BOOTSTRAP_SEED, "unit": "held-out observation record"},
        "interpretation_boundary": "P3-C is a broad-extrapolation stress diagnostic. Any relative advantage does not establish reliable reconstruction of the held-out region, physical validity of the PDE, or a general advantage of zero-wind B2.",
    }
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_metrics(comparison)
    plot_bootstrap(bootstrap)
    plot_diffusivity(summary)
    print("B2 P3-C analysis completed")
    print(comparison.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
