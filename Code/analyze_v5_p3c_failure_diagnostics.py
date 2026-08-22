"""Diagnose the error structure of frozen P3-C stress-test predictions.

This script is read-only with respect to trained models. It aggregates the
five-seed B1 and M1-r predictions, attaches training-record distance measures,
and reports descriptive error structure. The output is not a causal test of
the PDE or of the physical validity of the PMIP4 forcing.
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
PROTOCOL = ROOT / "Data" / "model_results" / "v5_pmip4_2d" / "p3_protocol"
OUTPUT = ROOT / "Data" / "model_results" / "v5_pmip4_2d" / "p3c_failure_diagnostics"
FIGURES = ROOT / "Figures"

SPLITS = ["stress_leave_south", "stress_leave_tropics", "stress_leave_north_excluding_p2dev"]
SPLIT_LABELS = {
    "stress_leave_south": "Southern high latitudes",
    "stress_leave_tropics": "Tropical belt",
    "stress_leave_north_excluding_p2dev": "Northern high latitudes",
}
MODELS = ["B1_pmip4_constd", "M1r_pmip4_relativeD"]
MODEL_LABELS = {"B1_pmip4_constd": "Wind-PINN", "M1r_pmip4_relativeD": "Relative-D PINN"}
MODEL_COLORS = {"B1_pmip4_constd": "#2a6f97", "M1r_pmip4_relativeD": "#c65d3b"}
SEEDS = [42, 520, 2002, 2026, 3407]
EARTH_RADIUS_KM = 6371.0088


def haversine_km(test_lon: np.ndarray, test_lat: np.ndarray, train_lon: np.ndarray, train_lat: np.ndarray) -> np.ndarray:
    lon1 = np.deg2rad(test_lon)[:, None]
    lat1 = np.deg2rad(test_lat)[:, None]
    lon2 = np.deg2rad(train_lon)[None, :]
    lat2 = np.deg2rad(train_lat)[None, :]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_ensemble(split: str, model: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        matches = list(RUNS.glob(f"p3c_{split}_{'b1_constd' if model == 'B1_pmip4_constd' else 'm1_relative'}_seed{seed}/**/test_predictions.csv"))
        if len(matches) != 1:
            raise RuntimeError(f"{split}/{model}/seed{seed}: expected one prediction file, found {len(matches)}")
        frames.append(pd.read_csv(matches[0]).set_index("record_id").sort_index())
    reference = frames[0][["lon", "lat", "log_dep"]].copy()
    for frame in frames[1:]:
        if not reference.index.equals(frame.index) or not np.allclose(reference.log_dep, frame.log_dep):
            raise RuntimeError(f"{split}/{model}: held-out records differ across seeds")
    prediction_matrix = np.column_stack([frame.prediction_log_dep.to_numpy(dtype=float) for frame in frames])
    reference["prediction_mean"] = prediction_matrix.mean(axis=1)
    reference["prediction_seed_std"] = prediction_matrix.std(axis=1, ddof=1)
    reference["error"] = reference.prediction_mean - reference.log_dep
    reference["abs_error"] = reference.error.abs()
    return reference.reset_index()


def attach_training_geometry(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    train = pd.read_csv(PROTOCOL / split / "train_records.csv")
    distances = haversine_km(
        frame.lon.to_numpy(dtype=float), frame.lat.to_numpy(dtype=float),
        train.lon.to_numpy(dtype=float), train.lat.to_numpy(dtype=float),
    )
    ordered = np.sort(distances, axis=1)
    output = frame.copy()
    output["nearest_train_km"] = ordered[:, 0]
    output["fifth_nearest_train_km"] = ordered[:, min(4, ordered.shape[1] - 1)]
    output["train_records_within_1000km"] = (distances <= 1000.0).sum(axis=1)
    output["train_records_within_2000km"] = (distances <= 2000.0).sum(axis=1)
    output["split"] = split
    output["n_train_records"] = len(train)
    return output


def build_record_table() -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for model in MODELS:
            rows.append(attach_training_geometry(load_ensemble(split, model), split).assign(model=model))
    table = pd.concat(rows, ignore_index=True)
    if not np.isfinite(table[["prediction_mean", "abs_error", "nearest_train_km", "fifth_nearest_train_km"]].to_numpy()).all():
        raise RuntimeError("Non-finite diagnostic values")
    return table


def summarize_records(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split, model), frame in records.groupby(["split", "model"], sort=False):
        nearest_spearman = frame.abs_error.corr(frame.nearest_train_km, method="spearman")
        fifth_spearman = frame.abs_error.corr(frame.fifth_nearest_train_km, method="spearman")
        median_distance = frame.nearest_train_km.median()
        near = frame[frame.nearest_train_km <= median_distance]
        far = frame[frame.nearest_train_km > median_distance]
        rows.append({
            "split": split,
            "split_label": SPLIT_LABELS[split],
            "model": model,
            "model_label": MODEL_LABELS[model],
            "n_test_records": len(frame),
            "mean_error_bias": frame.error.mean(),
            "mae_ensemble": frame.abs_error.mean(),
            "rmse_ensemble": float(np.sqrt(np.mean(frame.error.to_numpy() ** 2))),
            "median_nearest_train_km": median_distance,
            "median_fifth_nearest_train_km": frame.fifth_nearest_train_km.median(),
            "spearman_abs_error_vs_nearest_train_km": nearest_spearman,
            "spearman_abs_error_vs_fifth_nearest_train_km": fifth_spearman,
            "near_half_mae": near.abs_error.mean(),
            "far_half_mae": far.abs_error.mean(),
            "far_minus_near_mae": far.abs_error.mean() - near.abs_error.mean(),
            "mean_seed_prediction_std": frame.prediction_seed_std.mean(),
        })
    return pd.DataFrame(rows)


def build_distance_bins(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split, model), frame in records.groupby(["split", "model"], sort=False):
        labels = ["Nearer half", "Farther half"]
        bins = pd.qcut(frame.nearest_train_km, q=2, labels=labels, duplicates="drop")
        for label, group in frame.assign(distance_bin=bins).groupby("distance_bin", observed=True):
            rows.append({
                "split": split,
                "split_label": SPLIT_LABELS[split],
                "model": model,
                "model_label": MODEL_LABELS[model],
                "distance_bin": str(label),
                "n_records": len(group),
                "mean_nearest_train_km": group.nearest_train_km.mean(),
                "mae": group.abs_error.mean(),
                "bias": group.error.mean(),
            })
    return pd.DataFrame(rows)


def save_distance_figure(records: pd.DataFrame, bins: pd.DataFrame) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(7.0, 6.2), constrained_layout=True)
    for row, split in enumerate(SPLITS):
        scatter_axis, bar_axis = axes[row]
        for model in MODELS:
            frame = records[(records.split == split) & (records.model == model)]
            scatter_axis.scatter(
                frame.nearest_train_km, frame.abs_error, label=MODEL_LABELS[model],
                color=MODEL_COLORS[model], alpha=0.76, edgecolors="#343a40", linewidths=0.25,
            )
        scatter_axis.set(
            title=SPLIT_LABELS[split], xlabel="Distance to nearest training record (km)",
            ylabel="Absolute error (log deposition)",
        )
        scatter_axis.grid(alpha=0.25, linewidth=0.5)
        if row == 0:
            scatter_axis.legend(frameon=False, ncol=1, loc="lower right")

        subset = bins[bins.split == split]
        positions = np.arange(2)
        width = 0.32
        for index, model in enumerate(MODELS):
            model_rows = subset[subset.model == model].set_index("distance_bin").loc[["Nearer half", "Farther half"]]
            bar_axis.bar(
                positions + (index - 0.5) * width, model_rows.mae, width=width,
                color=MODEL_COLORS[model], label=MODEL_LABELS[model],
            )
        bar_axis.set(
            xticks=positions, xticklabels=["Nearer half", "Farther half"],
            ylabel="Ensemble MAE (log deposition)", title=f"{SPLIT_LABELS[split]} distance strata",
        )
        bar_axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    save_figure_all_formats(FIGURES / "V5_P3C_FAILURE_DISTANCE_DIAGNOSTIC", figure)
    plt.close(figure)


def save_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    display = summary[[
        "split_label", "model_label", "n_test_records", "mean_error_bias", "mae_ensemble",
        "median_nearest_train_km", "spearman_abs_error_vs_nearest_train_km", "far_minus_near_mae",
    ]].copy()
    display.columns = ["Stress region", "Model", "N", "Bias", "MAE", "Median nearest km", "Spearman rho", "Far-near MAE"]
    for column in ["Bias", "MAE", "Median nearest km", "Spearman rho", "Far-near MAE"]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    figure, axis = plt.subplots(figsize=(11.6, 3.6))
    axis.axis("off")
    table = axis.table(cellText=display.values, colLabels=display.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#b0b0b0")
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#e9ecef")
            cell.set_text_props(weight="bold")
        elif row % 2 == 1:
            cell.set_facecolor("#f7f7f7")
    axis.set_title("P3-C error structure: ensemble diagnostics relative to the split training geometry", pad=12)
    save_figure_all_formats(FIGURES / "V5_P3C_FAILURE_DIAGNOSTICS_TABLE", figure)
    plt.close(figure)
    return display


def write_manifest(records: pd.DataFrame, summary: pd.DataFrame) -> None:
    manifest = {
        "purpose": "Post hoc descriptive diagnosis of frozen P3-C predictions; no training, retuning, or causal PDE test.",
        "models": MODELS,
        "seeds": SEEDS,
        "splits": SPLITS,
        "record_rows": int(len(records)),
        "summary_rows": int(len(summary)),
        "distance_definition": "Great-circle haversine distance from each held-out record to P3 split training records.",
        "distance_strata": "Within each split/model, nearer and farther halves defined by the median nearest-training-record distance.",
    }
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    apply_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = build_record_table()
    summary = summarize_records(records)
    bins = build_distance_bins(records)
    records.to_csv(OUTPUT / "p3c_failure_record_diagnostics.csv", index=False)
    summary.to_csv(OUTPUT / "p3c_failure_summary.csv", index=False)
    bins.to_csv(OUTPUT / "p3c_failure_distance_bins.csv", index=False)
    save_distance_figure(records, bins)
    display = save_summary_table(summary)
    display.to_csv(OUTPUT / "p3c_failure_display_table.csv", index=False)
    write_manifest(records, summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
