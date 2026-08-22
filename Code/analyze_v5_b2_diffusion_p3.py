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
B1_RUNS = V5 / "p3_runs"
MAP_METRICS = V5 / "pde_map" / "p3_analysis" / "p3_pde_map_metrics.csv"
OUTPUT = V5 / "b2_diffusion_only" / "p3_analysis"
FIGURES = ROOT / "Figures"
SPLITS = ["outer_north_america", "outer_east_asia", "outer_tropical_atlantic_africa"]
SPLIT_LABELS = ["North America", "East Asia", "Tropical Atlantic-Africa"]
SEEDS = [42, 520, 2002, 2026, 3407]
N_TEST = {"outer_north_america": 64, "outer_east_asia": 106, "outer_tropical_atlantic_africa": 42}
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
    for run in sorted(B2_RUNS.glob("p3b_outer_*_b2_diffusion_seed*")):
        metrics_path = run / "b2_metrics.json"
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        split, seed = result["split"], int(result["seed"])
        pred_path = run / split / "B2_diffusion_only_PINN" / "test_predictions.csv"
        frame = pd.read_csv(pred_path)
        result["run_id"] = run.name
        result["metrics_path"] = str(metrics_path.relative_to(ROOT)).replace("\\", "/")
        rows.append(result)
        predictions[(split, seed)] = frame
    return pd.DataFrame(rows), predictions


def load_b1() -> dict[tuple[str, int], pd.DataFrame]:
    outputs = {}
    for path in B1_RUNS.glob("p3b_outer_*_b1_constd_seed*/**/test_predictions.csv"):
        result = json.loads(path.with_name("summary.json").read_text(encoding="utf-8"))
        outputs[(result["split"], int(result["seed"]))] = pd.read_csv(path)
    return outputs


def load_nopde() -> dict[tuple[str, int], pd.DataFrame]:
    outputs = {}
    for path in B1_RUNS.glob("p3b_outer_*_b2_nopde_seed*/**/test_predictions.csv"):
        result = json.loads(path.with_name("summary.json").read_text(encoding="utf-8"))
        outputs[(result["split"], int(result["seed"]))] = pd.read_csv(path)
    return outputs


def validate(
    b2_metrics: pd.DataFrame,
    b2_predictions: dict[tuple[str, int], pd.DataFrame],
    b1_predictions: dict[tuple[str, int], pd.DataFrame],
    nopde_predictions: dict[tuple[str, int], pd.DataFrame],
) -> None:
    expected = {(split, seed) for split in SPLITS for seed in SEEDS}
    actual = set(zip(b2_metrics.split, b2_metrics.seed.astype(int)))
    if actual != expected:
        raise RuntimeError(f"B2 P3-B matrix mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    if not (b2_metrics.epochs_requested.eq(150000).all() and b2_metrics.epochs_completed.eq(150000).all()):
        raise RuntimeError("At least one B2 task did not complete 150k epochs")
    if not b2_metrics.prediction_finite_fraction.eq(1.0).all():
        raise RuntimeError("At least one B2 task has non-finite predictions")
    if set(b2_predictions) != expected or set(b1_predictions) != expected or set(nopde_predictions) != expected:
        raise RuntimeError("B1/B2/no-PDE prediction matrix is incomplete")
    required = {"record_id", "lon", "lat", "log_dep", "prediction_log_dep"}
    for key, frame in {**b2_predictions, **b1_predictions, **nopde_predictions}.items():
        missing = required - set(frame.columns)
        if missing or len(frame) != N_TEST[key[0]] or frame.record_id.duplicated().any():
            raise RuntimeError(f"{key}: invalid held-out prediction frame")
        if not np.isfinite(frame.prediction_log_dep.to_numpy(dtype=float)).all():
            raise RuntimeError(f"{key}: non-finite predictions")
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
    result = metrics.groupby("split", sort=False)[columns].agg(["mean", "std", "min", "max", "count"]).reset_index()
    result.columns = ["_".join(filter(None, value)).rstrip("_") for value in result.columns.to_flat_index()]
    return result


def compare_models(
    b1_predictions: dict[tuple[str, int], pd.DataFrame],
    b2_predictions: dict[tuple[str, int], pd.DataFrame],
    nopde_predictions: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    map_frame = pd.read_csv(MAP_METRICS)
    rows = []
    for split, label in zip(SPLITS, SPLIT_LABELS):
        for name, predictions in [
            ("B1 PMIP4 PINN", b1_predictions),
            ("B2 zero-wind PINN", b2_predictions),
            ("B2 no-PDE MLP", nopde_predictions),
        ]:
            frame = ensemble(split, predictions)
            values = raw_metrics(frame.log_dep.to_numpy(dtype=float), frame.prediction_mean.to_numpy(dtype=float))
            rows.append({"split": split, "split_label": label, "model": name, "n_test": len(frame), **values})
        for name in ["PDE-MAP", "B3 GP/RBF"]:
            matches = map_frame[(map_frame["split"] == split) & (map_frame["model"] == name)]
            if len(matches) != 1:
                raise RuntimeError(f"{split}/{name}: expected one reference metric row")
            row = matches.iloc[0]
            rows.append({"split": split, "split_label": label, "model": name, "n_test": int(row.n_test), **{key: float(row[key]) for key in ["rmse", "mae", "r2", "bias", "p95_abs_error"]}})
    return pd.DataFrame(rows)


def bootstrap_candidate_vs_reference(
    reference_name: str,
    reference_predictions: dict[tuple[str, int], pd.DataFrame],
    candidate_name: str,
    candidate_predictions: dict[tuple[str, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary, draws = [], []
    for split in SPLITS:
        reference = ensemble(split, reference_predictions).set_index("record_id")
        candidate = ensemble(split, candidate_predictions).set_index("record_id")
        if not reference.index.equals(candidate.index) or not np.allclose(reference.log_dep, candidate.log_dep):
            raise RuntimeError(f"{split}: {reference_name}/{candidate_name} predictions cannot be paired")
        observed = reference.log_dep.to_numpy(dtype=float)
        reference_error = reference.prediction_mean.to_numpy(dtype=float) - observed
        candidate_error = candidate.prediction_mean.to_numpy(dtype=float) - observed
        indices = rng.integers(0, len(observed), size=(N_BOOTSTRAP, len(observed)))
        for metric, values in [
            ("rmse", np.sqrt(np.mean(candidate_error[indices] ** 2, axis=1)) - np.sqrt(np.mean(reference_error[indices] ** 2, axis=1))),
            ("mae", np.mean(np.abs(candidate_error[indices]) , axis=1) - np.mean(np.abs(reference_error[indices]), axis=1)),
        ]:
            observed_delta = float(np.sqrt(np.mean(candidate_error**2)) - np.sqrt(np.mean(reference_error**2))) if metric == "rmse" else float(np.mean(np.abs(candidate_error)) - np.mean(np.abs(reference_error)))
            summary.append({
                "split": split,
                "metric": metric,
                "n_holdout_records": len(observed),
                "n_bootstrap": N_BOOTSTRAP,
                "delta_definition": f"{candidate_name} ensemble metric minus {reference_name} ensemble metric; positive favors {reference_name}",
                "observed_delta": observed_delta,
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "bootstrap_fraction_delta_gt_zero": float(np.mean(values > 0.0)),
            })
            draws.append(pd.DataFrame({"split": split, "metric": metric, "bootstrap_delta": values}))
    return pd.DataFrame(summary), pd.concat(draws, ignore_index=True)


def plot_metrics(comparison: pd.DataFrame) -> None:
    colors = ["#2a6f97", "#c65d3b", "#6c757d", "#5d9c59", "#e0a100"]
    models = ["B1 PMIP4 PINN", "B2 zero-wind PINN", "B2 no-PDE MLP", "PDE-MAP", "B3 GP/RBF"]
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharey=True, constrained_layout=True)
    for axis, split, label in zip(axes, SPLITS, SPLIT_LABELS):
        subset = comparison[comparison.split == split].set_index("model").loc[models]
        axis.bar(range(len(models)), subset.rmse, color=colors, width=0.68)
        axis.set(title=label, xticks=range(len(models)), xticklabels=["B1", "B2-z", "MLP", "MAP", "GP"])
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Held-out ensemble RMSE (log deposition)")
    figure.suptitle("Frozen P3-B comparison: zero-wind ablation and reference baselines", y=1.03)
    save_figure_all_formats(FIGURES / "V5_B2_P3B_RMSE_COMPARISON", figure)
    plt.close(figure)


def plot_bootstrap(bootstrap: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    labels = ["N. America", "E. Asia", "Trop. Atl.-Afr."]
    for axis, metric, title in zip(axes, ["rmse", "mae"], ["RMSE difference", "MAE difference"]):
        frame = bootstrap[bootstrap.metric == metric].set_index("split").loc[SPLITS]
        values = frame.observed_delta.to_numpy(dtype=float)
        low, high = values - frame.ci95_low.to_numpy(dtype=float), frame.ci95_high.to_numpy(dtype=float) - values
        axis.errorbar(range(3), values, yerr=np.vstack([low, high]), fmt="none", ecolor="#343a40", capsize=3, linewidth=0.8)
        axis.scatter(range(3), values, s=30, color="#c65d3b", zorder=3)
        axis.axhline(0.0, color="#343a40", linewidth=0.8)
        axis.set(xticks=range(3), xticklabels=labels, title=title, ylabel="B2 zero-wind minus B1 PMIP4-wind")
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    figure.suptitle("P3-B paired held-out-record bootstrap, 95% CI; positive favors B1", y=1.03)
    save_figure_all_formats(FIGURES / "V5_B2_P3B_VS_B1_BOOTSTRAP", figure)
    plt.close(figure)


def plot_diffusivity(summary: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    values = summary.set_index("split").loc[SPLITS]
    scale = 1e6
    centers = (values.d_nd_constant_mean.to_numpy(dtype=float) - 0.02) * scale
    low = centers - (values.d_nd_constant_min.to_numpy(dtype=float) - 0.02) * scale
    high = (values.d_nd_constant_max.to_numpy(dtype=float) - 0.02) * scale - centers
    axis.errorbar(range(3), centers, yerr=np.vstack([low, high]), fmt="o", color="#c65d3b", capsize=4)
    axis.axhline(0.0, color="#343a40", linestyle="--", linewidth=0.9, label="parameter lower bound")
    axis.set(
        xticks=range(3),
        xticklabels=SPLIT_LABELS,
        ylabel=r"B2 learned $D_{nd} - 0.02$ ($10^{-6}$)",
        title="Zero-wind B2 diffusivity collapses to its lower bound",
    )
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False)
    save_figure_all_formats(FIGURES / "V5_B2_P3B_DIFFUSIVITY_BOUND", figure)
    plt.close(figure)


def main() -> None:
    apply_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    b2_metrics, b2_predictions = load_b2()
    b1_predictions = load_b1()
    nopde_predictions = load_nopde()
    validate(b2_metrics, b2_predictions, b1_predictions, nopde_predictions)
    b2_metrics = b2_metrics.sort_values(["split", "seed"]).reset_index(drop=True)
    summary = b2_summary(b2_metrics)
    comparison = compare_models(b1_predictions, b2_predictions, nopde_predictions)
    bootstrap_b1, draws_b1 = bootstrap_candidate_vs_reference("B1 PMIP4-wind", b1_predictions, "B2 zero-wind", b2_predictions)
    bootstrap_nopde, draws_nopde = bootstrap_candidate_vs_reference("B2 no-PDE MLP", nopde_predictions, "B2 zero-wind", b2_predictions)
    bootstrap = pd.concat([bootstrap_b1.assign(comparison="B2 zero-wind vs B1"), bootstrap_nopde.assign(comparison="B2 zero-wind vs no-PDE")], ignore_index=True)
    draws = pd.concat([draws_b1.assign(comparison="B2 zero-wind vs B1"), draws_nopde.assign(comparison="B2 zero-wind vs no-PDE")], ignore_index=True)
    b2_metrics.to_csv(OUTPUT / "b2_p3b_metrics_all_seeds.csv", index=False)
    summary.to_csv(OUTPUT / "b2_p3b_metrics_summary.csv", index=False)
    comparison.to_csv(OUTPUT / "b2_p3b_ensemble_reference_comparison.csv", index=False)
    bootstrap.to_csv(OUTPUT / "b2_p3b_vs_b1_observation_bootstrap.csv", index=False)
    draws.to_csv(OUTPUT / "b2_p3b_vs_b1_observation_bootstrap_draws.csv", index=False)
    manifest = {
        "purpose": "Frozen P3-B analysis of the B2 diffusion-only ablation; no model training or hyperparameter selection.",
        "formal_b2_runs": int(len(b2_metrics)),
        "splits": SPLITS,
        "seeds": SEEDS,
        "epochs_required": 150000,
        "bootstrap": {"n_replicates": N_BOOTSTRAP, "rng_seed": BOOTSTRAP_SEED, "unit": "held-out observation record"},
        "key_interpretation_boundary": "A B1 advantage over zero-wind B2 supports incremental predictive information from the wind terms under this frozen setup; it does not validate the PDE as a complete physical transport model or establish PINN necessity relative to PDE-MAP.",
    }
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_metrics(comparison)
    plot_bootstrap(bootstrap[bootstrap.comparison == "B2 zero-wind vs B1"])
    plot_diffusivity(summary)
    print("B2 P3-B analysis completed")
    print(comparison.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
