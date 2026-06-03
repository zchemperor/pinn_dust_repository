import numpy as np
import pandas as pd
from scipy import stats
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_multi_seed import aggregate_training_points

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..')
RESULTS = os.path.join(ROOT, 'Data', 'model_results')


def bootstrap_ci(diffs, n_boot=10000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.array([
        rng.choice(diffs, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, [alpha * 100, (1 - alpha) * 100])


def main():
    orig_dir = os.getcwd()
    os.chdir(ROOT)

    df_baseline, summary_bl = aggregate_training_points(
        "df_pinn_empirical_LGM_Baseline_training_points_seed*.csv",
        "df_pinn_empirical_LGM_Baseline_seed_summary.csv"
    )

    os.chdir(orig_dir)

    if df_baseline is None:
        print("[ERROR] No Baseline seed files found!")
        return
    
    v3_path = os.path.join(RESULTS, "df_pinn_empirical_LGM_seed_summary.csv")
    df_v3 = pd.read_csv(v3_path)

    print(f"\nv3 seeds: {sorted(df_v3['seed'].tolist())}")
    print(f"Baseline seeds: {sorted(df_baseline['seed'].tolist())}")

    merged = df_v3.merge(df_baseline, on='seed', suffixes=('_v3', '_bl'))
    print(f"Matched seeds: {len(merged)}")

    rows = []
    for metric in ['R2', 'RMSE', 'MAE']:
        v3_vals = merged[f'{metric}_v3'].values
        bl_vals = merged[f'{metric}_bl'].values
        diffs = v3_vals - bl_vals

        t_stat, t_p = stats.ttest_rel(v3_vals, bl_vals)

        try:
            w_stat, w_p = stats.wilcoxon(v3_vals, bl_vals)
        except ValueError:
            w_stat, w_p = np.nan, np.nan

        ci_lo, ci_hi = bootstrap_ci(diffs)

        row = {
            'metric': metric,
            'v3_mean': np.mean(v3_vals),
            'v3_std': np.std(v3_vals, ddof=1),
            'baseline_mean': np.mean(bl_vals),
            'baseline_std': np.std(bl_vals, ddof=1),
            'diff_mean': np.mean(diffs),
            'diff_std': np.std(diffs, ddof=1),
            'ttest_t': t_stat,
            'ttest_p': t_p,
            'wilcoxon_stat': w_stat,
            'wilcoxon_p': w_p,
            'bootstrap_ci_lower': ci_lo,
            'bootstrap_ci_upper': ci_hi,
            'n_seeds': len(diffs),
        }
        rows.append(row)

        sign = '+' if np.mean(diffs) > 0 else ''

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS, 'paired_test_v3_vs_baseline.csv')
    df_out.to_csv(out_path, index=False)

    for _, r in merged.iterrows():
        d = r['R2_v3'] - r['R2_bl']

if __name__ == '__main__':
    main()
