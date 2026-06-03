import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_jcp import apply_style, SAVE_DPI

apply_style()

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..')
RESULTS = os.path.join(ROOT, 'Data', 'model_results')
FIGURES = os.path.join(ROOT, 'Figures')

# Colors
C_V3 = '#2166ac'       # blue
C_BL = '#b2182b'       # red
C_HOL = '#4daf4a'      # green
C_LGM = '#984ea3'      # purple


def save_fig(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGURES, f'{name}.{ext}'), dpi=SAVE_DPI)
    print(f'  [SAVED] {name}.pdf/.png')


def plot_paired_test():
    print('\n--- Figure A: Paired Test ---')
    df = pd.read_csv(os.path.join(RESULTS, 'paired_test_v3_vs_baseline.csv'))

    # Also load per-seed data to plot individual points
    v3_sum = pd.read_csv(os.path.join(RESULTS, 'df_pinn_empirical_LGM_seed_summary.csv'))
    bl_sum = pd.read_csv(os.path.join(RESULTS, 'df_pinn_empirical_LGM_Baseline_seed_summary.csv'))
    merged = v3_sum.merge(bl_sum, on='seed', suffixes=('_v3', '_bl'))

    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.8))

    for i, metric in enumerate(['R2', 'RMSE', 'MAE']):
        ax = axes[i]
        row = df[df['metric'] == metric].iloc[0]

        diffs = merged[f'{metric}_v3'].values - merged[f'{metric}_bl'].values

        # Individual seed differences
        ax.scatter(range(len(diffs)), diffs, color=C_V3, s=30, zorder=5,
                   label='Per-seed diff')

        # Mean line
        ax.axhline(row['diff_mean'], color=C_V3, ls='--', lw=0.8, alpha=0.7)

        # Bootstrap CI band
        ax.axhspan(row['bootstrap_ci_lower'], row['bootstrap_ci_upper'],
                    alpha=0.15, color=C_V3, label='Bootstrap 95% CI')

        # Zero line
        ax.axhline(0, color='k', ls='-', lw=0.5)

        # Formatting
        ax.set_xlabel('Seed index')
        if metric == 'R2':
            ax.set_ylabel(r'$\Delta R^2$ (v3 $-$ Baseline)')
            ax.set_title(r'$R^2$')
        elif metric == 'RMSE':
            ax.set_ylabel(r'$\Delta$RMSE')
            ax.set_title('RMSE')
        else:
            ax.set_ylabel(r'$\Delta$MAE')
            ax.set_title('MAE')

        ax.set_xticks(range(len(diffs)))
        ax.set_xticklabels([str(s) for s in sorted(merged['seed'].tolist())],
                           rotation=45, fontsize=7)

        # p-value annotation
        p = row['ttest_p']
        p_str = f'p={p:.3f}' if p >= 0.001 else f'p={p:.1e}'
        ax.text(0.95, 0.95, p_str, transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

        if i == 0:
            ax.legend(fontsize=7, loc='lower left')

    fig.tight_layout()
    save_fig(fig, 'PAIRED_TEST_v3_vs_Baseline')
    plt.close(fig)


def plot_spatial_cv_bands():
    print('\n--- Figure B: Spatial CV Bands ---')
    df = pd.read_csv(os.path.join(RESULTS, 'spatial_cv_summary.csv'))

    bands = ['South', 'Tropical', 'North', 'Global']
    x = np.arange(len(bands))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7.48, 3.5))

    for j, (model, color, label) in enumerate([
        ('v3', C_V3, 'v3 (D(x))'),
        ('Baseline', C_BL, 'Baseline (constD)')
    ]):
        r2_means = []
        r2_stds = []
        for band in bands:
            row = df[(df['model'] == model) & (df['band'] == band)]
            r2_means.append(row['R2_mean'].values[0])
            r2_stds.append(row['R2_std'].values[0])

        offset = (j - 0.5) * w
        bars = ax.bar(x + offset, r2_means, w, yerr=r2_stds,
                      color=color, alpha=0.8, label=label,
                      capsize=3, error_kw={'lw': 0.8})

    # Add N labels
    for i, band in enumerate(bands):
        n = int(df[(df['model'] == 'v3') & (df['band'] == band)]['N'].values[0])
        ax.text(i, 0.72, f'N={n}', ha='center', fontsize=7, color='gray')

    ax.set_ylabel(r'$R^2$')
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylim(0.70, 1.0)
    ax.legend(loc='upper right')
    ax.set_title('v3 vs Baseline: Latitude-Band Analysis (Emp-LGM, 5-seed)')

    fig.tight_layout()
    save_fig(fig, 'SPATIAL_CV_BAND_COMPARISON')
    plt.close(fig)


def load_loss(filename):
    path = os.path.join(RESULTS, filename)
    if not os.path.exists(path):
        return None, None
    d = np.loadtxt(path)
    steps = d[:, 0]
    total_loss = d[:, 1]  # column 1 = total train loss
    return steps, total_loss


def plot_loss_composite():
    print('\n--- Figure C: Loss Composite ---')

    panels = [
        ('Emp-Holocene (Teacher)', 'loss_history_model_empirical_Holocene.dat', None),
        ('Emp-LGM: v3 vs Baseline', 'loss_history_model_empirical_LGM.dat',
         'loss_history_model_empirical_LGM_Baseline.dat'),
        ('Sim-Holocene (Teacher)', 'loss_history_model_simulated_Holocene.dat', None),
        ('Sim-LGM: v3 vs Baseline', 'loss_history_model_simulated_LGM.dat',
         'loss_history_model_simulated_LGM_Baseline.dat'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.48, 5.0))
    axes = axes.flatten()

    for i, (title, v3_file, bl_file) in enumerate(panels):
        ax = axes[i]
        steps, loss_v3 = load_loss(v3_file)
        if steps is not None:
            ax.semilogy(steps / 1000, loss_v3, color=C_V3, lw=0.8,
                        label='v3 (D(x))')

        if bl_file:
            steps_bl, loss_bl = load_loss(bl_file)
            if steps_bl is not None:
                ax.semilogy(steps_bl / 1000, loss_bl, color=C_BL, lw=0.8,
                            label='Baseline (constD)')

        ax.set_xlabel('Step (k)')
        ax.set_ylabel('Total Loss')
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

    fig.tight_layout()
    save_fig(fig, 'LOSS_COMPOSITE')
    plt.close(fig)


def plot_ablation_chain():
    print('\n--- Figure D: Ablation Chain ---')
    df = pd.read_csv(os.path.join(RESULTS, 'ablation_table3_empirical_full_chain.csv'))

    versions = ['v0', 'v1', 'v2', 'v3']
    x = np.arange(len(versions))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7.48, 3.5))

    for j, (dataset, color, label) in enumerate([
        ('Emp-LGM', C_LGM, 'Emp-LGM (N=317)'),
        ('Emp-Hol', C_HOL, 'Emp-Hol (N=397)')
    ]):
        r2_vals = []
        for v in versions:
            row = df[(df['Config'] == v) & (df['Dataset'] == dataset)]
            r2_vals.append(row['R2'].values[0])

        offset = (j - 0.5) * w
        bars = ax.bar(x + offset, r2_vals, w, color=color, alpha=0.85,
                      label=label)

        # Value labels on bars
        for k, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{r2_vals[k]:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel(r'$R^2$')
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylim(0.84, 0.94)
    ax.legend(loc='upper left')
    ax.set_title('Ablation Chain: v0 (Baseline) to v3 (Full Model)')

    # Annotate innovations
    ax.annotate('+HybridNorm', xy=(0.5, 0.845), fontsize=7, ha='center', color='gray')
    ax.annotate('+Transfer', xy=(1.5, 0.845), fontsize=7, ha='center', color='gray')
    ax.annotate('+D(x)', xy=(2.5, 0.845), fontsize=7, ha='center', color='gray')

    fig.tight_layout()
    save_fig(fig, 'ABLATION_CHAIN_R2')
    plt.close(fig)


def plot_sensitivity_composite():
    """
    4 PDE weights × 2 periods.
    Layout: 4 rows (weights) × 4 cols (Hol map, Hol hist, LGM map, LGM hist).
    """
    print('\n--- Figure E: Sensitivity Composite ---')

    from style_jcp import load_world
    world = load_world()

    weights = [0.0, 0.1, 10.0, 1000.0]
    w_labels = ['0', '0.1', '10 (default)', '1000']
    periods = ['Holocene', 'LGM']

    # Load empirical data for scatter overlay
    emp = {}
    for p in periods:
        emp[p] = pd.read_csv(os.path.join(ROOT, 'Data', 'processed_data',
                                           f'df_empirical_{p}.csv'))

    # Load all sensitivity grids
    grids = {}
    for p in periods:
        grids[p] = []
        for i in range(4):
            df = pd.read_csv(os.path.join(RESULTS,
                             f'df_pinn_empirical_{p}_sensitivity_{i}.csv'))
            grids[p].append(df)

    fig, axes = plt.subplots(4, 4, figsize=(7.0, 8.5))
    # cols: Hol map | Hol hist | LGM map | LGM hist

    vlim = 3.2  # colorbar limit for log_dep

    for row, (w, wl) in enumerate(zip(weights, w_labels)):
        for col_offset, period in enumerate(periods):
            df_grid = grids[period][row]
            df_emp = emp[period]
            ax_map = axes[row, col_offset * 2]
            ax_hist = axes[row, col_offset * 2 + 1]

            # --- Map panel ---
            field = df_grid['PINN_log_dep'].values.reshape(60, 120)
            im = ax_map.imshow(field, origin='lower',
                               extent=[-180, 180, -90, 90],
                               cmap='viridis', vmin=-vlim, vmax=vlim)
            world.dissolve(by='continent').boundary.plot(
                ax=ax_map, color='black', linewidth=0.3)
            ax_map.scatter(df_emp['lon'], df_emp['lat'], c=df_emp['log_dep'],
                           cmap='viridis', vmin=-vlim, vmax=vlim,
                           s=4, edgecolors='k', linewidths=0.2, zorder=5)
            ax_map.set_xlim(-180, 180)
            ax_map.set_ylim(-90, 90)
            ax_map.set_xticks(np.arange(-180, 181, 90))
            ax_map.set_yticks(np.arange(-90, 91, 45))
            ax_map.tick_params(labelsize=7)

            if row == 0:
                ax_map.set_title(f'{period} Map', fontsize=9)
            if col_offset == 0:
                ax_map.set_ylabel(f'$w_{{PDE}}$={wl}', fontsize=8)

            # --- Histogram panel ---
            vals = df_grid['PINN_log_dep'].values
            ax_hist.hist(vals, bins=40, density=True, color='steelblue',
                         alpha=0.7, edgecolor='none')
            mu, std = np.mean(vals), np.std(vals)
            x_fit = np.linspace(-vlim, vlim, 200)
            from scipy.stats import norm as norm_dist
            ax_hist.plot(x_fit, norm_dist.pdf(x_fit, mu, std),
                         'k-', lw=0.8)
            ax_hist.set_xlim(-vlim, vlim)
            ax_hist.tick_params(labelsize=7)
            ax_hist.text(0.95, 0.92,
                         f'$\\mu$={mu:.2f}\n$\\sigma$={std:.2f}',
                         transform=ax_hist.transAxes, ha='right', va='top',
                         fontsize=7, bbox=dict(fc='white', ec='gray',
                                               boxstyle='round,pad=0.3',
                                               alpha=0.8))
            if row == 0:
                ax_hist.set_title(f'{period} Hist', fontsize=9)

            # Remove x labels except bottom row
            if row < 3:
                ax_map.set_xticklabels([])
                ax_hist.set_xticklabels([])

    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    save_fig(fig, 'SENSITIVITY_COMPOSITE')
    plt.close(fig)


def main():
    print('=' * 50)
    print('  Phase 4: Generate Paper Figures')
    print('=' * 50)

    plot_paired_test()
    plot_spatial_cv_bands()
    plot_loss_composite()
    plot_ablation_chain()
    plot_sensitivity_composite()

    print('\n[ALL DONE]')


if __name__ == '__main__':
    main()
