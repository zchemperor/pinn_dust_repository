import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_pde_residuals import (
    compute_pde_residual_Dx,
    compute_pde_residual_constD,
    reshape_to_grid,
    load_wind,
    read_D_from_variables,
)
from style_jcp import apply_style, SAVE_DPI, CMAP_DIFFUSIVITY, save_figure_all_formats

apply_style()

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..')
DATA = os.path.join(ROOT, 'Data')
RESULTS = os.path.join(DATA, 'model_results')
FIGURES = os.path.join(ROOT, 'Figures')

N_RANDOM = 20          # number of random D fields
SEED_BASE = 12345      # reproducibility


def generate_random_smooth_D(nlat, nlon, lats_rad, lons_rad,
                              D_mean, D_std, n_modes=8, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    lat_grid, lon_grid = np.meshgrid(lats_rad, lons_rad, indexing='ij')
    field = np.zeros((nlat, nlon))

    for _ in range(n_modes):
        # Random wavenumbers (low frequency: k in [1, 4])
        k_lat = rng.integers(1, 5)
        k_lon = rng.integers(1, 5)
        phase_lat = rng.uniform(0, 2 * np.pi)
        phase_lon = rng.uniform(0, 2 * np.pi)
        amplitude = rng.standard_normal()

        field += amplitude * (
            np.sin(k_lat * lat_grid + phase_lat) *
            np.sin(k_lon * lon_grid + phase_lon)
        )

    # Normalize to zero mean, unit std
    field = (field - field.mean()) / (field.std() + 1e-12)

    # Scale to match learned D statistics, then softplus for positivity
    # Use log-space matching: log(D_learned) has some mean/std
    # Map field -> softplus(field * scale + shift) to match D_mean, D_std
    raw = field * (D_std / D_mean) * 0.5 + np.log(np.exp(D_mean) - 1)
    D_random = np.log(1 + np.exp(raw))  # softplus

    return D_random


def run_null_test(period):
    print(f"\n{'='*60}")
    print(f"  Experiment A — v3 u-field: {period}")
    print(f"{'='*60}")

    # Load global grid
    grid_path = os.path.join(RESULTS,
                             f'df_pinn_empirical_{period}_global_grid.csv')
    df = pd.read_csv(grid_path)

    lons_deg = np.sort(df['lon'].unique())
    lats_deg = np.sort(df['lat'].unique())
    lons_rad = np.deg2rad(lons_deg)
    lats_rad = np.deg2rad(lats_deg)
    nlat, nlon = len(lats_deg), len(lons_deg)

    # Load u-field and learned D(x)
    u_grid = reshape_to_grid(df, 'PINN_log_dep', lats_deg, lons_deg)
    D_learned = reshape_to_grid(df, 'PINN_Diffusivity', lats_deg, lons_deg)

    # Load wind
    wind_interp = load_wind(DATA)
    K_vec = wind_interp(lats_deg / 90.0).astype(np.float64)

    # Compute learned D(x) PDE residual
    res_learned = compute_pde_residual_Dx(u_grid, D_learned,
                                           lats_rad, lons_rad, K_vec)
    learned_mean = np.mean(np.abs(res_learned))
    learned_median = np.median(np.abs(res_learned))
    learned_95 = np.percentile(np.abs(res_learned), 95)
    print(f"  Learned D(x): mean={learned_mean:.4f}, "
          f"median={learned_median:.4f}, 95%={learned_95:.4f}")

    # Statistics of learned D for matching
    D_mean = np.mean(D_learned)
    D_std = np.std(D_learned)
    print(f"  Learned D stats: mean={D_mean:.6f}, std={D_std:.6f}")

    # Generate random D fields and compute residuals
    random_means = []
    random_medians = []
    random_95s = []

    for i in range(N_RANDOM):
        rng = np.random.default_rng(SEED_BASE + i)
        D_random = generate_random_smooth_D(
            nlat, nlon, lats_rad, lons_rad,
            D_mean, D_std, n_modes=8, rng=rng
        )
        res_random = compute_pde_residual_Dx(u_grid, D_random,
                                              lats_rad, lons_rad, K_vec)
        r_mean = np.mean(np.abs(res_random))
        r_median = np.median(np.abs(res_random))
        r_95 = np.percentile(np.abs(res_random), 95)
        random_means.append(r_mean)
        random_medians.append(r_median)
        random_95s.append(r_95)

    random_means = np.array(random_means)
    print(f"  Random D (n={N_RANDOM}): mean={np.mean(random_means):.4f} "
          f"± {np.std(random_means):.4f}")

    # constD residual (use mean of learned D as scalar)
    res_constD = compute_pde_residual_constD(u_grid, D_mean,
                                              lats_rad, lons_rad, K_vec)
    constD_mean = np.mean(np.abs(res_constD))
    print(f"  Const D (D={D_mean:.6f}): mean={constD_mean:.4f}")

    return {
        'period': period,
        'learned_mean': learned_mean,
        'learned_median': learned_median,
        'learned_95': learned_95,
        'random_means': random_means,
        'random_medians': np.array(random_medians),
        'random_95s': np.array(random_95s),
        'constD_mean': constD_mean,
        'D_mean': D_mean,
        'D_std': D_std,
    }


def run_cross_comparison(period, D_mean_v3, D_std_v3):
    print(f"\n{'='*60}")
    print(f"  Experiment B — constD u-field: {period}")
    print(f"{'='*60}")

    # Load constD global grid (only LGM available)
    constD_grid_path = os.path.join(
        RESULTS, f'df_pinn_empirical_{period}_constD_global_grid.csv')
    if not os.path.exists(constD_grid_path):
        print(f"  [SKIP] constD grid not found: {constD_grid_path}")
        return None

    df_constD = pd.read_csv(constD_grid_path)
    lons_deg = np.sort(df_constD['lon'].unique())
    lats_deg = np.sort(df_constD['lat'].unique())
    lons_rad = np.deg2rad(lons_deg)
    lats_rad = np.deg2rad(lats_deg)
    nlat, nlon = len(lats_deg), len(lons_deg)

    u_constD = reshape_to_grid(df_constD, 'PINN_log_dep', lats_deg, lons_deg)

    wind_interp = load_wind(DATA)
    K_vec = wind_interp(lats_deg / 90.0).astype(np.float64)

    # Read the actual trained constD scalar value
    var_path = os.path.join(DATA, 'trained_models',
                            f'model_empirical_{period}_constD', 'variables.dat')
    D_scalar = read_D_from_variables(var_path)
    print(f"  Trained constD scalar: {D_scalar:.6f}")

    # Baseline: constD u-field + trained constD scalar
    res_baseline = compute_pde_residual_constD(
        u_constD, D_scalar, lats_rad, lons_rad, K_vec)
    baseline_mean = np.mean(np.abs(res_baseline))
    print(f"  constD u + trained scalar D: mean={baseline_mean:.4f}")

    # Also load v3's learned D(x) and pair with constD's u-field
    v3_grid_path = os.path.join(
        RESULTS, f'df_pinn_empirical_{period}_global_grid.csv')
    df_v3 = pd.read_csv(v3_grid_path)
    D_learned = reshape_to_grid(df_v3, 'PINN_Diffusivity', lats_deg, lons_deg)

    res_cross_learned = compute_pde_residual_Dx(
        u_constD, D_learned, lats_rad, lons_rad, K_vec)
    cross_learned_mean = np.mean(np.abs(res_cross_learned))
    print(f"  constD u + v3 learned D(x): mean={cross_learned_mean:.4f}")

    # Random D fields on constD's u-field
    random_means = []
    for i in range(N_RANDOM):
        rng = np.random.default_rng(SEED_BASE + i)
        D_random = generate_random_smooth_D(
            nlat, nlon, lats_rad, lons_rad,
            D_mean_v3, D_std_v3, n_modes=8, rng=rng
        )
        res_random = compute_pde_residual_Dx(
            u_constD, D_random, lats_rad, lons_rad, K_vec)
        random_means.append(np.mean(np.abs(res_random)))

    random_means = np.array(random_means)
    print(f"  constD u + random D (n={N_RANDOM}): "
          f"mean={np.mean(random_means):.4f} ± {np.std(random_means):.4f}")

    return {
        'period': period,
        'baseline_mean': baseline_mean,
        'cross_learned_mean': cross_learned_mean,
        'random_means': random_means,
        'D_scalar': D_scalar,
    }


def plot_combined(results_hol, results_lgm, cross_lgm):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.7))

    # --- Panels 0-1: Experiment A (v3 u-field) ---
    for ax, res, title in zip(axes[:2],
                               [results_hol, results_lgm],
                               ['(a) Exp A: Holocene', '(b) Exp A: LGM']):
        ax.hist(res['random_means'], bins=10, alpha=0.6,
                color='#888888', edgecolor='black', linewidth=0.5,
                label=f'Random D (n={N_RANDOM})', zorder=2)
        ax.axvline(res['learned_mean'], color='#d62728', linewidth=1.0,
                   linestyle='-', label='Learned D(x)', zorder=3)
        ax.axvline(res['constD_mean'], color='#1f77b4', linewidth=1.0,
                   linestyle='--', label='Constant D', zorder=3)
        ax.set_xlabel('Mean |PDE Residual|')
        ax.set_ylabel('Count')
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6, loc='upper right')

        ratio = np.mean(res['random_means']) / res['learned_mean']
        ax.text(0.05, 0.88,
                f"Learned: {res['learned_mean']:.2f}\n"
                f"Random: {np.mean(res['random_means']):.2f}\n"
                f"Ratio: {ratio:.1f}x",
                transform=ax.transAxes, fontsize=6.5,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='wheat', alpha=0.8))

    # --- Panel 2: Experiment B (constD u-field, LGM only) ---
    if cross_lgm is not None:
        ax = axes[2]
        ax.hist(cross_lgm['random_means'], bins=10, alpha=0.6,
                color='#888888', edgecolor='black', linewidth=0.5,
                label=f'Random D (n={N_RANDOM})', zorder=2)
        ax.axvline(cross_lgm['baseline_mean'], color='#1f77b4',
                   linewidth=1.0, linestyle='-',
                   label='Trained scalar D', zorder=3)
        ax.axvline(cross_lgm['cross_learned_mean'], color='#d62728',
                   linewidth=1.0, linestyle='--',
                   label="v3's D(x) (mismatched)", zorder=3)
        ax.set_xlabel('Mean |PDE Residual|')
        ax.set_ylabel('Count')
        ax.set_title('(c) Exp B: constD u-field, LGM', fontsize=9)
        ax.legend(fontsize=6, loc='upper right')

        ax.text(0.05, 0.88,
                f"Scalar D: {cross_lgm['baseline_mean']:.2f}\n"
                f"v3 D(x): {cross_lgm['cross_learned_mean']:.2f}\n"
                f"Random: {np.mean(cross_lgm['random_means']):.2f}",
                transform=ax.transAxes, fontsize=6.5,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='wheat', alpha=0.8))

    fig.tight_layout(rect=[0, 0.03, 1, 1], w_pad=0.5)
    save_figure_all_formats(os.path.join(FIGURES, 'NULL_TEST_RANDOM_D'),
                            fig=fig, dpi=SAVE_DPI, bbox_inches='tight')
    for ext in ['pdf', 'png', 'eps']:
        path = os.path.join(FIGURES, f'NULL_TEST_RANDOM_D.{ext}')
        print(f"[SAVED] {path}")
    plt.close(fig)


def save_results_csv(results_hol, results_lgm, cross_lgm):
    rows = []
    # Experiment A
    for res in [results_hol, results_lgm]:
        rows.append({
            'Experiment': 'A_v3_ufield',
            'Period': res['period'],
            'Learned_D_PDE_mean': res['learned_mean'],
            'Learned_D_PDE_median': res['learned_median'],
            'Learned_D_PDE_95pct': res['learned_95'],
            'Random_D_PDE_mean_avg': np.mean(res['random_means']),
            'Random_D_PDE_mean_std': np.std(res['random_means']),
            'Random_D_PDE_mean_min': np.min(res['random_means']),
            'Random_D_PDE_mean_max': np.max(res['random_means']),
            'ConstD_PDE_mean': res['constD_mean'],
            'Ratio_random_over_learned': (
                np.mean(res['random_means']) / res['learned_mean']
            ),
        })
    # Experiment B
    if cross_lgm is not None:
        rows.append({
            'Experiment': 'B_constD_ufield',
            'Period': cross_lgm['period'],
            'Learned_D_PDE_mean': cross_lgm['cross_learned_mean'],
            'Learned_D_PDE_median': np.nan,
            'Learned_D_PDE_95pct': np.nan,
            'Random_D_PDE_mean_avg': np.mean(cross_lgm['random_means']),
            'Random_D_PDE_mean_std': np.std(cross_lgm['random_means']),
            'Random_D_PDE_mean_min': np.min(cross_lgm['random_means']),
            'Random_D_PDE_mean_max': np.max(cross_lgm['random_means']),
            'ConstD_PDE_mean': cross_lgm['baseline_mean'],
            'Ratio_random_over_learned': (
                np.mean(cross_lgm['random_means']) /
                cross_lgm['cross_learned_mean']
            ),
        })
    df_out = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS, 'null_test_random_D.csv')
    df_out.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}")


if __name__ == '__main__':
    # Experiment A: v3 u-field with different D fields
    results_hol = run_null_test('Holocene')
    results_lgm = run_null_test('LGM')

    # Experiment B: constD u-field with random D fields (LGM only)
    cross_lgm = run_cross_comparison(
        'LGM', results_lgm['D_mean'], results_lgm['D_std'])

    # Summary
    print("\n" + "="*60)
    print("  Summary — Experiment A (v3 u-field)")
    print("="*60)
    for res in [results_hol, results_lgm]:
        ratio = np.mean(res['random_means']) / res['learned_mean']
        print(f"  {res['period']:10s}: Learned={res['learned_mean']:.4f}, "
              f"Random={np.mean(res['random_means']):.4f} ({ratio:.1f}x), "
              f"ConstD={res['constD_mean']:.4f}")

    if cross_lgm is not None:
        print(f"\n  Summary — Experiment B (constD u-field, LGM)")
        print(f"  {'':10s}  Scalar D={cross_lgm['baseline_mean']:.4f}, "
              f"v3 D(x)={cross_lgm['cross_learned_mean']:.4f}, "
              f"Random={np.mean(cross_lgm['random_means']):.4f}")

    plot_combined(results_hol, results_lgm, cross_lgm)
    save_results_csv(results_hol, results_lgm, cross_lgm)
    print("\n[DONE] Null test complete!")
