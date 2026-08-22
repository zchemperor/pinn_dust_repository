"""Analyze the completed minimal V5 PDE-loss-weight robustness audit.

This is a post hoc robustness analysis of fixed P3-C stress splits. It reads
frozen run outputs only and does not choose a new loss weight or train models.
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
AUDIT = V5 / "loss_weight_robustness_audit"
PROTOCOL = AUDIT / "audit_protocol.json"
FORMAL_RUNS = AUDIT / "formal_runs"
OUTPUT = AUDIT / "analysis"
FIGURES = ROOT / "Figures"
SPLITS = ["stress_leave_south", "stress_leave_tropics", "stress_leave_north_excluding_p2dev"]
SPLIT_LABELS = ["Southern high latitudes", "Tropical belt", "Northern high latitudes"]
SEEDS = [42, 520, 2002]
WEIGHTS = [0.0, 0.3, 1.0, 3.0]
N_TEST = {"stress_leave_south": 36, "stress_leave_tropics": 70, "stress_leave_north_excluding_p2dev": 175}
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 20260807


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weight_tag(weight: float) -> str:
    return f"w{weight:g}".replace(".", "p")


def raw_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    return {
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "bias": float(error.mean()),
        "p95_abs_error": float(np.percentile(np.abs(error), 95)),
    }


def valid_prediction(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"record_id", "lon", "lat", "log_dep", "prediction_log_dep"}
    if required - set(frame.columns) or len(frame) != N_TEST[split] or frame.record_id.duplicated().any():
        raise RuntimeError(f"{path}: invalid held-out prediction frame")
    if not np.isfinite(frame.prediction_log_dep.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{path}: non-finite held-out prediction")
    if "p2_development_record" in frame and frame.p2_development_record.astype(bool).any():
        raise RuntimeError(f"{path}: P2 development record leaked into the test output")
    return frame


def load_existing_nopde() -> dict[tuple[float, str, int], pd.DataFrame]:
    predictions = {}
    for run in sorted((V5 / "p3_runs").glob("p3c_stress_*_b2_nopde_seed*")):
        summary_paths = list(run.glob("*/B2_nopde_mlp/summary.json"))
        if len(summary_paths) != 1:
            raise RuntimeError(f"{run.name}: expected one no-PDE summary")
        result = json.loads(summary_paths[0].read_text(encoding="utf-8"))
        split, seed = result["split"], int(result["seed"])
        if seed in SEEDS:
            predictions[(0.0, split, seed)] = valid_prediction(summary_paths[0].with_name("test_predictions.csv"), split)
    return predictions


def load_existing_diffusion() -> dict[tuple[float, str, int], pd.DataFrame]:
    predictions = {}
    for run in sorted((V5 / "b2_diffusion_only" / "p3_runs").glob("p3c_stress_*_b2_diffusion_seed*")):
        result = json.loads((run / "b2_metrics.json").read_text(encoding="utf-8"))
        split, seed = result["split"], int(result["seed"])
        if seed in SEEDS:
            path = run / split / "B2_diffusion_only_PINN" / "test_predictions.csv"
            predictions[(1.0, split, seed)] = valid_prediction(path, split)
    return predictions


def load_audit_runs(protocol_hash: str) -> tuple[pd.DataFrame, dict[tuple[float, str, int], pd.DataFrame]]:
    rows, predictions = [], {}
    for run in sorted(FORMAL_RUNS.glob("formal_w*_stress_*_seed*")):
        metadata = json.loads((run / "experiment_protocol.json").read_text(encoding="utf-8"))
        if metadata.get("audit_protocol_sha256") != protocol_hash:
            raise RuntimeError(f"{run.name}: mismatched audit protocol hash")
        result = json.loads((run / "audit_metrics.json").read_text(encoding="utf-8"))
        weight, split, seed = float(result["pde_residual_weight"]), result["split"], int(result["seed"])
        model = f"V5_loss_weight_audit_{weight_tag(weight)}"
        path = run / split / model / "test_predictions.csv"
        result["run_id"] = run.name
        result["metrics_path"] = str((run / "audit_metrics.json").relative_to(ROOT)).replace("\\", "/")
        rows.append(result)
        predictions[(weight, split, seed)] = valid_prediction(path, split)
    return pd.DataFrame(rows), predictions


def validate(audit_metrics: pd.DataFrame, predictions: dict[tuple[float, str, int], pd.DataFrame]) -> None:
    expected = {(weight, split, seed) for weight in [0.3, 3.0] for split in SPLITS for seed in SEEDS}
    actual = {(float(row.pde_residual_weight), row.split, int(row.seed)) for row in audit_metrics.itertuples()}
    if actual != expected:
        raise RuntimeError(f"Formal audit matrix mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    if not (audit_metrics.epochs_requested.eq(150000).all() and audit_metrics.epochs_completed.eq(150000).all()):
        raise RuntimeError("At least one formal audit task did not complete 150k epochs")
    if not audit_metrics.prediction_finite_fraction.eq(1.0).all():
        raise RuntimeError("At least one formal audit task has non-finite predictions")
    if set(predictions) != expected:
        raise RuntimeError("Formal audit prediction matrix is incomplete")


def ensemble(predictions: dict[tuple[float, str, int], pd.DataFrame], weight: float, split: str) -> pd.DataFrame:
    frames = [predictions[(weight, split, seed)].set_index("record_id") for seed in SEEDS]
    base = frames[0][["lon", "lat", "log_dep"]].copy()
    for frame in frames[1:]:
        if not base.index.equals(frame.index) or not np.allclose(base.log_dep, frame.log_dep):
            raise RuntimeError(f"weight={weight}, {split}: seed predictions use different records")
    matrix = np.column_stack([frame.prediction_log_dep.to_numpy(dtype=float) for frame in frames])
    base["prediction_mean"] = matrix.mean(axis=1)
    base["prediction_seed_sd"] = matrix.std(axis=1, ddof=1)
    return base.reset_index()


def comparison_table(predictions: dict[tuple[float, str, int], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split, label in zip(SPLITS, SPLIT_LABELS):
        for weight in WEIGHTS:
            frame = ensemble(predictions, weight, split)
            rows.append({
                "split": split,
                "split_label": label,
                "pde_residual_weight": weight,
                "n_test": len(frame),
                "mean_within_record_seed_sd": float(frame.prediction_seed_sd.mean()),
                **raw_metrics(frame.log_dep.to_numpy(dtype=float), frame.prediction_mean.to_numpy(dtype=float)),
            })
    return pd.DataFrame(rows)


def paired_bootstrap(predictions: dict[tuple[float, str, int], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries, draws = [], []
    for split in SPLITS:
        nopde = ensemble(predictions, 0.0, split).set_index("record_id")
        for weight in [0.3, 1.0, 3.0]:
            current = ensemble(predictions, weight, split).set_index("record_id")
            if not nopde.index.equals(current.index) or not np.allclose(nopde.log_dep, current.log_dep):
                raise RuntimeError(f"{split}: unable to pair weight 0 and {weight}")
            observed = nopde.log_dep.to_numpy(dtype=float)
            nopde_error = nopde.prediction_mean.to_numpy(dtype=float) - observed
            current_error = current.prediction_mean.to_numpy(dtype=float) - observed
            indices = rng.integers(0, len(observed), size=(N_BOOTSTRAP, len(observed)))
            for metric, values in [
                ("rmse", np.sqrt(np.mean(nopde_error[indices] ** 2, axis=1)) - np.sqrt(np.mean(current_error[indices] ** 2, axis=1))),
                ("mae", np.mean(np.abs(nopde_error[indices]), axis=1) - np.mean(np.abs(current_error[indices]), axis=1)),
            ]:
                observed_delta = (
                    float(np.sqrt(np.mean(nopde_error**2)) - np.sqrt(np.mean(current_error**2)))
                    if metric == "rmse"
                    else float(np.mean(np.abs(nopde_error)) - np.mean(np.abs(current_error)))
                )
                summaries.append({
                    "split": split,
                    "pde_residual_weight": weight,
                    "metric": metric,
                    "n_holdout_records": len(observed),
                    "n_bootstrap": N_BOOTSTRAP,
                    "delta_definition": "no-PDE MLP ensemble metric minus zero-wind diffusion PINN ensemble metric; positive favors diffusion PINN",
                    "observed_delta": observed_delta,
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "bootstrap_fraction_delta_gt_zero": float(np.mean(values > 0.0)),
                })
                draws.append(pd.DataFrame({"split": split, "pde_residual_weight": weight, "metric": metric, "bootstrap_delta": values}))
    return pd.DataFrame(summaries), pd.concat(draws, ignore_index=True)


def plot_rmse(comparison: pd.DataFrame) -> None:
    colors = ["#6c757d", "#77a8b7", "#2a6f97", "#c65d3b"]
    labels = ["0\n(no PDE)", "0.3", "1\n(nominal)", "3"]
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 3.35), sharey=True, constrained_layout=True)
    for axis, split, title in zip(axes, SPLITS, SPLIT_LABELS):
        frame = comparison[comparison.split == split].set_index("pde_residual_weight").loc[WEIGHTS]
        axis.plot(range(len(WEIGHTS)), frame.rmse, marker="o", color="#343a40", linewidth=1.1, zorder=2)
        axis.scatter(range(len(WEIGHTS)), frame.rmse, c=colors, s=34, zorder=3)
        axis.set(title=title, xticks=range(len(WEIGHTS)), xticklabels=labels, xlabel="PDE residual weight")
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Three-seed ensemble RMSE (log deposition)")
    save_figure_all_formats(FIGURES / "V5_LOSS_WEIGHT_AUDIT_RMSE", figure)
    plt.close(figure)


def plot_bootstrap(bootstrap: pd.DataFrame) -> None:
    labels = ["0.3", "1", "3"]
    colors = ["#77a8b7", "#2a6f97", "#c65d3b"]
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 3.3), sharey=True, constrained_layout=True)
    for axis, split, title in zip(axes, SPLITS, SPLIT_LABELS):
        frame = bootstrap[(bootstrap.split == split) & (bootstrap.metric == "rmse")].set_index("pde_residual_weight").loc[[0.3, 1.0, 3.0]]
        values = frame.observed_delta.to_numpy(dtype=float)
        lower = values - frame.ci95_low.to_numpy(dtype=float)
        upper = frame.ci95_high.to_numpy(dtype=float) - values
        axis.errorbar(range(3), values, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#343a40", capsize=3, linewidth=0.8)
        axis.scatter(range(3), values, color=colors, s=32, zorder=3)
        axis.axhline(0.0, color="#343a40", linewidth=0.8)
        axis.set(title=title, xticks=range(3), xticklabels=labels, xlabel="PDE residual weight")
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("No-PDE MLP RMSE minus\nDiffusion-PINN RMSE (log deposition)")
    save_figure_all_formats(FIGURES / "V5_LOSS_WEIGHT_AUDIT_VS_NOPDE_BOOTSTRAP", figure)
    plt.close(figure)


def main() -> None:
    apply_style()
    protocol_hash = sha256(PROTOCOL)
    audit_metrics, audit_predictions = load_audit_runs(protocol_hash)
    validate(audit_metrics, audit_predictions)
    predictions = {**load_existing_nopde(), **audit_predictions, **load_existing_diffusion()}
    expected_all = {(weight, split, seed) for weight in WEIGHTS for split in SPLITS for seed in SEEDS}
    if set(predictions) != expected_all:
        raise RuntimeError(f"Reference matrix mismatch; missing={sorted(expected_all - set(predictions))}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison = comparison_table(predictions)
    bootstrap, draws = paired_bootstrap(predictions)
    audit_metrics.sort_values(["pde_residual_weight", "split", "seed"]).to_csv(OUTPUT / "formal_audit_metrics_all_seeds.csv", index=False)
    comparison.to_csv(OUTPUT / "three_seed_ensemble_weight_comparison.csv", index=False)
    bootstrap.to_csv(OUTPUT / "paired_bootstrap_vs_nopde.csv", index=False)
    draws.to_csv(OUTPUT / "paired_bootstrap_vs_nopde_draws.csv", index=False)
    manifest = {
        "purpose": "Post hoc P3-C robustness audit of PDE-residual weights 0.3 and 3.0 around existing weights 0.0 and 1.0.",
        "audit_protocol": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "audit_protocol_sha256": protocol_hash,
        "formal_tasks": int(len(audit_metrics)),
        "weights": WEIGHTS,
        "splits": SPLITS,
        "seeds": SEEDS,
        "bootstrap": {"n_replicates": N_BOOTSTRAP, "rng_seed": BOOTSTRAP_SEED, "unit": "held-out observation record"},
        "interpretation_boundary": "No result may select a preferred loss weight or be described as prospective confirmation, calibrated uncertainty, physical validation, or global reconstruction evidence.",
    }
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_rmse(comparison)
    plot_bootstrap(bootstrap)
    print(comparison.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
