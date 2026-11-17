"""P4: compare frozen predictions with explicit constant baselines; no training.

Writes a new audit directory only. No imports from training/plotting modules.
All metrics use unstandardized log10 deposition and record-level equal weights.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "Data/model_results/v5_pmip4_2d"
OUTPUT = V5 / "constant_baseline_audit_20261114"
SEEDS = [42, 520, 2002, 2026, 3407]
N_BOOTSTRAP = 20_000
RNG_SEED = 20261114
SPLITS = {
    "outer_north_america": "北美确认区",
    "outer_east_asia": "东亚确认区",
    "outer_tropical_atlantic_africa": "热带 Atlantic–Africa 确认区",
    "stress_leave_south": "南部广泛外推区",
    "stress_leave_tropics": "热带广泛外推区",
    "stress_leave_north_excluding_p2dev": "北部广泛外推区（保留开发块）",
}
NEURAL = {
    "Wind-PINN": ("b1_constd", "B1_pmip4_constd"),
    "Diffusion-PINN": ("b2_diffusion", "B2_diffusion_only_PINN"),
    "Relative-D": ("m1_relative", "M1r_pmip4_relativeD"),
    "No-PDE MLP": ("b2_nopde", "B2_nopde_mlp"),
}
METRICS = ["rmse", "mae", "r2", "bias", "p95_abs_error"]
PROTECTED = [
    ROOT / "manuscript/main.tex",
    ROOT / "manuscript/cas-refs.bib",
    ROOT / "manuscript/main.pdf",
    ROOT / "manuscript/main_anon.pdf",
    ROOT / "manuscript/submission/01_Manuscript_anonymized.pdf",
    # One internal revision-plan entry is omitted from this release.
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y
    mse = np.mean(error**2)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1 - mse / np.mean((y - y.mean())**2)),
        "bias": float(error.mean()),
        "p95_abs_error": float(np.percentile(np.abs(error), 95)),
    }


def prediction_path(split: str, model: str, seed: int) -> Path:
    stage = "p3b" if split.startswith("outer_") else "p3c"
    suffix, folder = NEURAL[model]
    base = V5 / ("b2_diffusion_only/p3_runs" if model == "Diffusion-PINN" else "p3_runs")
    return base / f"{stage}_{split}_{suffix}_seed{seed}" / split / folder / "test_predictions.csv"


def bootstrap_deltas(y, matrix, baseline, indices, estimand):
    """Paired record bootstrap conditional on fixed training mean and fitted models.

    For mean_seed_metric: calculate each seed's metric in every resample, then
    average metrics; never substitute the metric of averaged predictions.
    """
    error = matrix - y[:, None]
    baseline_error = baseline - y
    rmse, mae = [], []
    for start in range(0, len(indices), 500):
        idx = indices[start:start + 500]
        sampled = error[idx]
        model_rmse = np.sqrt(np.mean(sampled**2, axis=1))
        model_mae = np.mean(np.abs(sampled), axis=1)
        if estimand == "mean_seed_metric":
            model_rmse = model_rmse.mean(axis=1)
            model_mae = model_mae.mean(axis=1)
        else:
            model_rmse, model_mae = model_rmse[:, 0], model_mae[:, 0]
        base = baseline_error[idx]
        rmse.append(model_rmse - np.sqrt(np.mean(base**2, axis=1)))
        mae.append(model_mae - np.mean(np.abs(base), axis=1))
    return {"rmse": np.concatenate(rmse), "mae": np.concatenate(mae)}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite an existing audit: {OUTPUT}")
    protected_before = {str(p.relative_to(ROOT)): sha256(p) for p in PROTECTED}
    sources = {}

    def read_csv(path):
        sources[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)
        return pd.read_csv(path)

    metric_rows, per_seed_rows, summary_rows, delta_rows, record_frames = [], [], [], [], []
    split_rows, availability, validations = [], [], []
    rng = np.random.default_rng(RNG_SEED)
    for split, label in SPLITS.items():
        protocol = V5 / "p3_protocol" / split
        train = read_csv(protocol / "train_records.csv")
        test = read_csv(protocol / "test_records.csv").set_index("record_id").sort_index()
        assert not test.index.has_duplicates and not train.record_id.duplicated().any()
        assert not set(train.record_id) & set(test.index)
        for frame in [train, test]:
            assert np.isfinite(frame[["lon", "lat", "log_dep"]].to_numpy()).all()
        y = test.log_dep.to_numpy(float)
        n = len(y)
        train_mean, test_mean = float(train.log_dep.mean()), float(y.mean())
        metadata_path = protocol / "manifest.json"
        sources[str(metadata_path.relative_to(ROOT)).replace("\\", "/")] = sha256(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        np.testing.assert_allclose(train_mean, metadata["normalization"]["mean"], atol=1e-12, rtol=0)
        np.testing.assert_allclose((y - train_mean) / train.log_dep.std(ddof=1), test.log_dep_norm, atol=1e-12, rtol=0)
        constants = {"Train-mean constant": train_mean, "Test-mean oracle": test_mean}
        baseline_metrics = metrics(y, np.full(n, train_mean))
        split_rows.append({"split": split, "split_label": label, "n_train": len(train), "n_test": n,
                           "n_train_coordinates": len(train.drop_duplicates(["lon", "lat"])),
                           "n_test_coordinates": len(test.drop_duplicates(["lon", "lat"])),
                           "train_mean_log10": train_mean, "test_mean_log10": test_mean,
                           "test_sd_population_log10": float(y.std(ddof=0))})
        records = test[["lon", "lat", "log_dep"]].copy()
        records.insert(0, "split", split)
        for model, value in constants.items():
            pred = np.full(n, value)
            m = metrics(y, pred)
            if model == "Test-mean oracle":
                np.testing.assert_allclose(m["r2"], 0, atol=1e-14)
                np.testing.assert_allclose(m["rmse"]**2, np.mean((y - y.mean())**2), atol=1e-14)
            metric_rows.append({"split": split, "split_label": label, "model": model,
                                "estimand": "test_label_oracle" if "oracle" in model else "deterministic",
                                "n_test": n, "n_seeds": 0, **m})
            records[model] = pred

        matrices = {}

        def load_aligned(path):
            f = read_csv(path).set_index("record_id").sort_index()
            assert f.index.equals(test.index) and not f.index.has_duplicates, str(path)
            np.testing.assert_allclose(f[["lon", "lat", "log_dep"]], test[["lon", "lat", "log_dep"]], atol=1e-12, rtol=0)
            pred = f.prediction_log_dep.to_numpy(float)
            assert np.isfinite(pred).all(), str(path)
            return pred

        for model in NEURAL:
            columns, seed_metrics = [], []
            for seed in SEEDS:
                pred = load_aligned(prediction_path(split, model, seed))
                columns.append(pred)
                m = metrics(y, pred)
                seed_metrics.append(m)
                per_seed_rows.append({"split": split, "split_label": label, "model": model,
                                      "seed": seed, "n_test": n, **m})
            matrix = np.column_stack(columns)
            matrices[model] = matrix
            summary = pd.DataFrame(seed_metrics)
            row = {"split": split, "split_label": label, "model": model, "n_seeds": len(SEEDS), "n_test": n}
            for metric in METRICS:
                row[metric + "_mean"] = float(summary[metric].mean())
                row[metric + "_sd"] = float(summary[metric].std(ddof=1))
            row["n_seeds_lower_rmse_than_train_mean"] = int((summary.rmse < baseline_metrics["rmse"]).sum())
            row["n_seeds_lower_mae_than_train_mean"] = int((summary.mae < baseline_metrics["mae"]).sum())
            summary_rows.append(row)
            ensemble = matrix.mean(axis=1)
            metric_rows.append({"split": split, "split_label": label, "model": model,
                                "estimand": "five_seed_ensemble", "n_test": n, "n_seeds": len(SEEDS), **metrics(y, ensemble)})
            records[model] = ensemble

        static = {"PDE-MAP": V5 / "pde_map" / f"{'p3' if split.startswith('outer_') else 'p3c'}_formal_{split}" / "test_predictions.csv"}
        gp = V5 / "p3_runs" / f"{'p3b' if split.startswith('outer_') else 'p3c'}_{split}_b3_gp_seed42" / split / "B3_gp_rbf/test_predictions.csv"
        availability.append({"split": split, "model": "GP/RBF", "available": gp.exists(),
                             "reason": "existing formal fit" if gp.exists() else "no existing formal stress GP prediction; no refit performed"})
        if gp.exists():
            static["GP/RBF"] = gp
        for model, path in static.items():
            pred = load_aligned(path)
            matrices[model] = pred[:, None]
            metric_rows.append({"split": split, "split_label": label, "model": model,
                                "estimand": "deterministic", "n_test": n, "n_seeds": 0, **metrics(y, pred)})
            records[model] = pred

        indices = rng.integers(0, n, size=(N_BOOTSTRAP, n))
        for model, matrix in matrices.items():
            estimands = {"five_seed_ensemble": matrix.mean(axis=1)[:, None], "mean_seed_metric": matrix} if model in NEURAL else {"deterministic": matrix}
            for estimand, values in estimands.items():
                if estimand == "mean_seed_metric":
                    point = {k: float(np.mean([metrics(y, values[:, j])[k] for j in range(values.shape[1])])) for k in ["rmse", "mae"]}
                else:
                    point = metrics(y, values[:, 0])
                for metric, draws in bootstrap_deltas(y, values, train_mean, indices, estimand).items():
                    low, high = np.quantile(draws, [0.025, 0.975])
                    delta_rows.append({"split": split, "split_label": label, "model": model,
                                       "estimand": estimand, "metric": metric,
                                       "delta_model_minus_train_mean": point[metric] - baseline_metrics[metric],
                                       "ci95_low": float(low), "ci95_high": float(high),
                                       "interpretation": "model_lower_error" if high < 0 else "model_higher_error" if low > 0 else "interval_crosses_zero",
                                       "n_test": n, "n_bootstrap": N_BOOTSTRAP})
        record_frames.append(records.reset_index())
        validations.append({"split": split, "aligned_prediction_files": 20 + len(static),
                            "checks": "unique/disjoint IDs; identical test IDs/targets/coordinates; finite values; normalization matches frozen manifest; oracle R2=0"})

    all_metrics = pd.DataFrame(metric_rows)
    per_seed = pd.DataFrame(per_seed_rows)
    summary = pd.DataFrame(summary_rows)
    deltas = pd.DataFrame(delta_rows)
    old = read_csv(V5 / "wind_vs_zerowind_audit/ensemble_metrics.csv")
    mapping = {"wind": "Wind-PINN", "zerowind": "Diffusion-PINN", "nopde": "No-PDE MLP"}
    max_difference = 0.0
    for _, row in old.iterrows():
        actual = all_metrics[(all_metrics.split == row.split) & (all_metrics.model == mapping[row.model])].iloc[0]
        for key in METRICS:
            difference = abs(float(actual[key]) - float(row[key]))
            max_difference = max(max_difference, difference)
            np.testing.assert_allclose(actual[key], row[key], atol=1e-11, rtol=0)
    protected_after = {str(p.relative_to(ROOT)): sha256(p) for p in PROTECTED}
    assert protected_before == protected_after, "Protected input files changed"

    manifest = {
        "purpose": "Constant-baseline audit of frozen predictions; no training, tuning, or manuscript changes",
        "seeds": SEEDS, "target": "unstandardized log10 deposition, reference unit g m^-2 year^-1",
        "weighting": "equal weight per held-out record, retaining existing duplicates",
        "train_mean": "mean of frozen training log_dep for each split; fixed in bootstrap",
        "test_mean_oracle": "test-label-dependent descriptive R2 reference; not deployable and not bootstrapped",
        "bootstrap": {"draws": N_BOOTSTRAP, "seed": RNG_SEED, "unit": "paired held-out record", "interval": "percentile 95%, unadjusted for multiple comparisons", "reuse": "same record indices within split for all models/estimands", "limitations": "conditional on fixed fitted models and training mean; excludes training sampling, seed resampling, colocated/spatial dependence and archive error"},
        "delta_definition": "model error minus train-mean baseline error; negative favors model",
        "gp_stress": "No existing formal predictions found; not fitted for this audit",
        "prior_ensemble_max_abs_metric_difference": max_difference,
        "validated_prediction_files": sum(v["aligned_prediction_files"] for v in validations),
        "protected_files_sha256_before": protected_before, "protected_files_sha256_after": protected_after,
        "source_sha256": sources, "script_sha256": sha256(Path(__file__)), "validation": validations,
    }
    report = make_report(all_metrics, summary, deltas, manifest)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    all_metrics.to_csv(OUTPUT / "model_and_constant_metrics.csv", index=False)
    per_seed.to_csv(OUTPUT / "neural_metrics_by_seed.csv", index=False)
    summary.to_csv(OUTPUT / "neural_seed_summary.csv", index=False)
    deltas.to_csv(OUTPUT / "paired_bootstrap_vs_train_mean.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUTPUT / "split_baseline_details.csv", index=False)
    pd.DataFrame(availability).to_csv(OUTPUT / "model_availability.csv", index=False)
    pd.concat(record_frames, ignore_index=True).to_csv(OUTPUT / "aligned_record_predictions.csv", index=False)
    (OUTPUT / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT / "P4_常数基线核查报告.md").write_text(report, encoding="utf-8")
    print(all_metrics.pivot(index="split_label", columns="model", values="rmse").round(4).to_string())
    print("\nWind-PINN paired RMSE contrasts (negative favors Wind-PINN):")
    print(deltas[(deltas.model == "Wind-PINN") & (deltas.metric == "rmse")].to_string(index=False))
    print("\nOUTPUT:", OUTPUT)
    print("Protected manuscript/plan hashes unchanged; legacy max metric difference:", max_difference)


def make_report(results, summary, deltas, manifest):
    lines = ["# 冻结预测与常数基线核查报告", "", "补充核查；未训练、未调整模型，未改动既有材料。", "",
             "## 1．分析口径", "",
             "全部指标在未标准化的 log10 沉积量尺度计算，参考单位为 g·m⁻²·年⁻¹。训练集均值由每个冻结划分的训练 log_dep 计算，属于可实施的常数预测；测试集均值使用测试标签，仅是 R² 的事后参考，不是可部署预测器。其 RMSE 最小化性质不意味着它也是 MAE 最优常数（MAE 对应中位数）。",
             "", "四个神经模型各保留五个正式种子，并分别评价逐种子指标均值和五种子平均预测。GP/RBF、PDE-MAP 评价已有确定性预测；没有补拟合广泛外推的 GP。", "",
             "## 2．最终预测器的 RMSE", "", "神经模型列为五种子平均预测后的指标；其他列为确定性结果。空缺表示无既有正式预测，不能解释为性能差。", "",
             "| 区域 | 训练均值 | 测试均值参考 | Wind-PINN | Diffusion-PINN | Relative-D | No-PDE MLP | GP/RBF | PDE-MAP |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    models = ["Train-mean constant", "Test-mean oracle", "Wind-PINN", "Diffusion-PINN", "Relative-D", "No-PDE MLP", "GP/RBF", "PDE-MAP"]
    for split, label in SPLITS.items():
        f = results[results.split == split].set_index("model")
        lines.append("| " + label + " | " + " | ".join(f"{f.loc[m, 'rmse']:.4f}" if m in f.index else "—" for m in models) + " |")
    lines += ["", "## 3．Wind-PINN 相对训练均值的配对 RMSE 差", "", "差值为模型减常数基线，负值有利于模型。区间来自 20,000 次配对记录级 bootstrap。", "",
              "| 区域 | 五种子集成差 [95%区间] | 逐种子RMSE均值差 [95%区间] | 单种子RMSE低于常数的个数 |", "|---|---|---|---:|"]
    for split, label in SPLITS.items():
        parts = []
        for estimand in ["five_seed_ensemble", "mean_seed_metric"]:
            r = deltas[(deltas.split == split) & (deltas.model == "Wind-PINN") & (deltas.estimand == estimand) & (deltas.metric == "rmse")].iloc[0]
            parts.append(f"{r.delta_model_minus_train_mean:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]")
        s = summary[(summary.split == split) & (summary.model == "Wind-PINN")].iloc[0]
        lines.append(f"| {label} | {' | '.join(parts)} | {s.n_seeds_lower_rmse_than_train_mean}/5 |")
    lines += ["", "## 4．逐种子 RMSE 均值 ± SD", "", "| 区域 | Wind-PINN | Diffusion-PINN | Relative-D | No-PDE MLP |", "|---|---:|---:|---:|---:|"]
    for split, label in SPLITS.items():
        s = summary[summary.split == split].set_index("model")
        lines.append("| " + label + " | " + " | ".join(f"{s.loc[m, 'rmse_mean']:.4f} ± {s.loc[m, 'rmse_sd']:.4f}" for m in NEURAL) + " |")
    lines += ["", "## 5．各区域判断（最终预测器口径）", ""]
    for split, label in SPLITS.items():
        f = results[results.split == split].set_index("model")
        candidates = f.drop(index=["Test-mean oracle"])
        winner = candidates.rmse.idxmin()
        wind = f.loc["Wind-PINN"]
        r = deltas[(deltas.split == split) & (deltas.model == "Wind-PINN") & (deltas.estimand == "five_seed_ensemble") & (deltas.metric == "rmse")].iloc[0]
        judgment = {"model_lower_error": "Wind-PINN 的 RMSE 低于训练均值，名义95%区间不跨零", "model_higher_error": "Wind-PINN 的 RMSE 高于训练均值，名义95%区间不跨零", "interval_crosses_zero": "Wind-PINN 与训练均值的 RMSE 差区间跨零，不能称为明确胜出"}[r.interpretation]
        lines.append(f"- **{label}**：{judgment}；Wind-PINN R²={wind.r2:.4f}。已有可实施预测器中点估计 RMSE 最低为 {winner}（{candidates.loc[winner, 'rmse']:.4f}）。")
    lines += ["", "## 6．解释边界与复核", "",
              "- 优于 No-PDE MLP、优于零风、优于训练均值和 R²>0 是四个不同判断，不能替代。跨区域不做简单胜出计数来宣称普遍有效。",
              "- 正 R²表示相对测试均值参考的平方误差改善；训练均值比较还受到训练/测试分布差异影响。",
              "- 保留记录等权与现有重复条目，未完成档案身份或地点依赖修正。记录级区间可能受同址及空间相关性影响；不称为独立地点或总体泛化的确认。",
              "- 区间固定训练均值和已拟合模型；逐种子均值口径也不重采样种子，不能视为对任意随机种子总体的完整推断。",
              "- 区间为名义95%百分位区间，未作多重比较校正，不以小样本显著性代替实用效应或物理验证。",
              f"- 核验 {manifest['validated_prediction_files']} 个预测文件的记录ID、目标、坐标及有限值；复算既有18组 wind/zero/no-PDE集成指标的最大绝对差为 {manifest['prior_ensemble_max_abs_metric_difference']:.3g}。",
              "- 保护清单所列输入文件的前后 SHA-256 一致。",
              "", "## 7．文件说明与复现", "",
              "- `model_and_constant_metrics.csv`：全部模型最终预测器及两种常数参考的 RMSE、MAE、R²、bias、P95。",
              "- `neural_metrics_by_seed.csv` / `neural_seed_summary.csv`：单种子指标及均值、SD和胜过训练均值的种子数。",
              "- `paired_bootstrap_vs_train_mean.csv`：所有已有模型相对训练均值的RMSE/MAE差值与区间，神经模型分别列集成和逐种子指标均值。",
              "- `split_baseline_details.csv` / `model_availability.csv`：划分、常数、样本量和既有模型覆盖。",
              "- `aligned_record_predictions.csv`：按冻结 record_id 对齐的测试目标、常数和最终预测。",
              "- `audit_manifest.json`：来源哈希、分析定义、验证与受保护文件哈希。",
              "", "复现入口：`python -B Code/analyze_v5_constant_baseline_audit.py`。脚本拒绝覆盖已存在的审计目录；需要重复执行时应先明确另存输出位置，不删除现有证据。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
