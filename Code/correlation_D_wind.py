
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr
import geopandas as gpd
from style_jcp import (apply_style, SAVE_DPI, DIFFUSIVITY_VLIM,
                        CMAP_DIFFUSIVITY, CMAP_DIVERGING, load_world)

apply_style()


BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..')
DATA = os.path.join(ROOT, 'Data')
RESULTS = os.path.join(DATA, 'model_results')
FIGURES = os.path.join(ROOT, 'Figures')
PROCESSED = os.path.join(DATA, 'processed_data')

PERIODS = {
    'Holocene': os.path.join(RESULTS, 'df_pinn_empirical_Holocene_global_grid.csv'),
    'LGM': os.path.join(RESULTS, 'df_pinn_empirical_LGM_global_grid.csv'),
}

REGIONS = {
    'Sahara':         {'lat': (15, 35),   'lon': (-20, 40)},
    'Gobi':           {'lat': (35, 50),   'lon': (75, 120)},
    'Patagonia':      {'lat': (-55, -40), 'lon': (-75, -60)},
    'Southern Ocean': {'lat': (-65, -45), 'lon': (-180, 180)},
    'Tropical':       {'lat': (-15, 15),  'lon': (-180, 180)},
}


def load_data():
    grids = {}
    for period, path in PERIODS.items():
        grids[period] = pd.read_csv(path)
    df_wind = pd.read_csv(os.path.join(PROCESSED, 'df_wind.csv'))
    return grids, df_wind


def compute_zonal_stats(df):
    grouped = df.groupby('lat')['PINN_Diffusivity']
    stats = grouped.agg(['mean', 'std']).reset_index()
    stats.columns = ['lat', 'zonal_mean_D', 'zonal_std_D']
    return stats


def interpolate_wind(df_wind, target_lats):
    f_wind = interp1d(df_wind['latitude'].values, df_wind['wind'].values,
                      kind='linear', fill_value='extrapolate')
    wind_interp = f_wind(target_lats)
    return wind_interp


def compute_correlations(zonal_D, abs_wind, signed_wind):
    r_pearson, p_pearson = pearsonr(zonal_D, abs_wind)
    r_spearman, p_spearman = spearmanr(zonal_D, abs_wind)
    r_signed, p_signed = pearsonr(zonal_D, signed_wind)
    return {
        'Pearson_r': r_pearson, 'Pearson_p': p_pearson,
        'Spearman_rho': r_spearman, 'Spearman_p': p_spearman,
        'Pearson_signed_r': r_signed, 'Pearson_signed_p': p_signed,
    }


def compute_regional_stats(grids, df_wind):
    rows = []
    for region_name, bounds in REGIONS.items():
        lat_min, lat_max = bounds['lat']
        lon_min, lon_max = bounds['lon']

        # 风场：纬度带内的平均 |wind|
        wind_mask = (df_wind['latitude'] >= lat_min) & (df_wind['latitude'] <= lat_max)
        mean_abs_wind = np.abs(df_wind.loc[wind_mask, 'wind']).mean()

        row = {'Region': region_name, 'lat_min': lat_min, 'lat_max': lat_max,
               'mean_abs_wind': mean_abs_wind}

        for period, df in grids.items():
            mask = (df['lat'] >= lat_min) & (df['lat'] <= lat_max)
            if lon_min != -180 or lon_max != 180:
                mask &= (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
            sub = df.loc[mask, 'PINN_Diffusivity']
            row[f'mean_D_{period}'] = sub.mean()
            row[f'std_D_{period}'] = sub.std()
            row[f'N_{period}'] = len(sub)

        rows.append(row)
    return pd.DataFrame(rows)



world = load_world()


def plot_zonal_profiles(zonal_all, wind_all, corr_all):
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.5))

    for ax, period in zip(axes, ['Holocene', 'LGM']):
        z = zonal_all[period]
        lats = z['lat'].values
        mean_D = z['zonal_mean_D'].values
        std_D = z['zonal_std_D'].values
        abs_w = wind_all[period]
        corr = corr_all[period]

        # D 曲线 (左轴)
        color_D = '#0070C0'
        ax.plot(lats, mean_D, '-', color=color_D, linewidth=1.2, label='Zonal mean D')
        ax.fill_between(lats, mean_D - std_D, mean_D + std_D,
                         color=color_D, alpha=0.15)
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Diffusivity Coefficient [-]', color=color_D)
        ax.tick_params(axis='y', labelcolor=color_D)
        ax.set_xlim(-90, 90)
        ax.set_ylim(0, DIFFUSIVITY_VLIM[1])

        # |wind| 曲线 (右轴)
        ax2 = ax.twinx()
        color_W = '#B00020'
        ax2.plot(lats, abs_w, '--', color=color_W, linewidth=1.0, label='|Wind|')
        ax2.set_ylabel('|Wind Speed| (m/s)', color=color_W)
        ax2.tick_params(axis='y', labelcolor=color_W)

        # 急流带标注
        for band in [(-60, -30), (30, 60)]:
            ax.axvspan(band[0], band[1], color='gray', alpha=0.08)

        # 统计标注
        txt = (f"Pearson r = {corr['Pearson_r']:.2f} (p = {corr['Pearson_p']:.1e})\n"
               f"Spearman ρ = {corr['Spearman_rho']:.2f} (p = {corr['Spearman_p']:.1e})")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4'))

        label = '(a)' if period == 'Holocene' else '(b)'
        ax.set_title(f'{label} {period}')

    fig.subplots_adjust(wspace=0.55)
    fig.savefig(os.path.join(FIGURES, 'CORRELATION_ZONAL_PROFILE_D_WIND.pdf'),
                bbox_inches='tight', dpi=SAVE_DPI)
    fig.savefig(os.path.join(FIGURES, 'CORRELATION_ZONAL_PROFILE_D_WIND.png'),
                bbox_inches='tight', dpi=SAVE_DPI)
    plt.close(fig)
    print('[FIG] CORRELATION_ZONAL_PROFILE_D_WIND')


def plot_scatter_D_wind(zonal_all, wind_all, corr_all):
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.5))

    for ax, period in zip(axes, ['Holocene', 'LGM']):
        z = zonal_all[period]
        lats = z['lat'].values
        mean_D = z['zonal_mean_D'].values
        abs_w = wind_all[period]
        corr = corr_all[period]

        sc = ax.scatter(abs_w, mean_D, c=lats, cmap='RdBu_r', s=30,
                        edgecolors='k', linewidths=0.3, vmin=-90, vmax=90)

        # 线性回归
        coeffs = np.polyfit(abs_w, mean_D, 1)
        x_fit = np.linspace(0, abs_w.max() * 1.05, 50)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), 'k--', linewidth=0.8)

        ax.set_xlabel('|Wind Speed| (m/s)')
        ax.set_ylabel('Zonal Mean D [-]')
        ax.set_ylim(0, DIFFUSIVITY_VLIM[1])

        txt = (f"Pearson r = {corr['Pearson_r']:.2f}\n"
               f"Spearman ρ = {corr['Spearman_rho']:.2f}")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4'))

        label = '(a)' if period == 'Holocene' else '(b)'
        ax.set_title(f'{label} {period}')

    cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label('Latitude')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'CORRELATION_SCATTER_D_WIND.pdf'),
                bbox_inches='tight', dpi=SAVE_DPI)
    fig.savefig(os.path.join(FIGURES, 'CORRELATION_SCATTER_D_WIND.png'),
                bbox_inches='tight', dpi=SAVE_DPI)
    plt.close(fig)
    print('[FIG] CORRELATION_SCATTER_D_WIND')


def plot_map_overlay(grids, df_wind):
    fig, axes = plt.subplots(2, 1, figsize=(7.48, 8.0))

    for ax, period in zip(axes, ['Holocene', 'LGM']):
        df = grids[period]
        n_lat = len(df['lat'].unique())
        n_lon = len(df['lon'].unique())
        D_field = df['PINN_Diffusivity'].values.reshape(n_lat, n_lon)

        im = ax.imshow(D_field, origin='lower', extent=[-180, 180, -90, 90],
                       cmap=CMAP_DIFFUSIVITY,
                       vmin=DIFFUSIVITY_VLIM[0], vmax=DIFFUSIVITY_VLIM[1],
                       aspect='auto')

        world.dissolve(by='continent').boundary.plot(ax=ax, color='white',
                                                      linewidth=0.5)

        # 叠加风场剖面（缩放到经度空间）
        wind_scaled = df_wind['wind'].values
        wind_max = np.abs(wind_scaled).max()
        wind_x = -180 + (wind_scaled / wind_max) * 30  # 缩放到 30° 宽度
        ax.plot(wind_x, df_wind['latitude'].values, 'w-', linewidth=0.8, alpha=0.9)
        ax.axvline(-180, color='white', linewidth=0.3, alpha=0.5)

        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(-180, 181, 45))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        label = '(a)' if period == 'Holocene' else '(b)'
        ax.set_title(f'{label} {period} — D(x) Field')

    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.04, pad=0.08)
    cbar.set_label('Diffusivity Coefficient [-]')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'CORRELATION_MAP_D_WIND_OVERLAY.pdf'),
                bbox_inches='tight', dpi=SAVE_DPI)
    fig.savefig(os.path.join(FIGURES, 'CORRELATION_MAP_D_WIND_OVERLAY.png'),
                bbox_inches='tight', dpi=SAVE_DPI)
    plt.close(fig)
    print('[FIG] CORRELATION_MAP_D_WIND_OVERLAY')


def plot_regional_bars(df_regional):
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 4.0))

    for ax, period in zip(axes, ['Holocene', 'LGM']):
        regions = df_regional['Region'].values
        mean_D = df_regional[f'mean_D_{period}'].values
        std_D = df_regional[f'std_D_{period}'].values
        mean_wind = df_regional['mean_abs_wind'].values

        x = np.arange(len(regions))
        width = 0.35

        # D 柱
        color_D = '#0070C0'
        bars_D = ax.bar(x - width/2, mean_D, width, yerr=std_D,
                        color=color_D, alpha=0.8, label='Mean D', capsize=3)

        ax.set_ylabel('Diffusivity Coefficient [-]', color=color_D)
        ax.tick_params(axis='y', labelcolor=color_D)

        # |wind| 柱 (右轴)
        ax2 = ax.twinx()
        color_W = '#B00020'
        bars_W = ax2.bar(x + width/2, mean_wind, width,
                         color=color_W, alpha=0.6, label='Mean |Wind|')
        ax2.set_ylabel('|Wind Speed| (m/s)', color=color_W)
        ax2.tick_params(axis='y', labelcolor=color_W)

        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=30, ha='right')

        # 合并图例
        lines_D, labels_D = ax.get_legend_handles_labels()
        lines_W, labels_W = ax2.get_legend_handles_labels()
        ax.legend(lines_D + lines_W, labels_D + labels_W, loc='upper right')

        label = '(a)' if period == 'Holocene' else '(b)'
        ax.set_title(f'{label} {period}')

    fig.subplots_adjust(wspace=0.55, bottom=0.25)
    fig.savefig(os.path.join(FIGURES, 'CORRELATION_REGIONAL_D_ANALYSIS.pdf'),
                bbox_inches='tight', dpi=SAVE_DPI)
    fig.savefig(os.path.join(FIGURES, 'CORRELATION_REGIONAL_D_ANALYSIS.png'),
                bbox_inches='tight', dpi=SAVE_DPI)
    plt.close(fig)
    print('[FIG] CORRELATION_REGIONAL_D_ANALYSIS')



if __name__ == '__main__':
    grids, df_wind = load_data()

    zonal_all = {}
    wind_all = {}
    corr_all = {}
    summary_rows = []

    for period in ['Holocene', 'LGM']:
        df = grids[period]

        # 纬向统计
        zonal = compute_zonal_stats(df)

        # 排除极区 (lat=±90)
        zonal = zonal[(zonal['lat'] > -89) & (zonal['lat'] < 89)].copy()

        # 插值风场
        lats = zonal['lat'].values
        wind_interp = interpolate_wind(df_wind, lats)
        abs_wind = np.abs(wind_interp)

        # 相关性
        corr = compute_correlations(zonal['zonal_mean_D'].values, abs_wind, wind_interp)

        zonal_all[period] = zonal
        wind_all[period] = abs_wind
        corr_all[period] = corr

        for metric in ['Pearson_r', 'Pearson_p', 'Spearman_rho', 'Spearman_p',
                        'Pearson_signed_r', 'Pearson_signed_p']:
            summary_rows.append({
                'Period': period, 'Metric': metric, 'Value': corr[metric]
            })

        print(f"\n=== {period} ===")
        print(f"  Pearson r  = {corr['Pearson_r']:.3f} (p = {corr['Pearson_p']:.2e})")
        print(f"  Spearman ρ = {corr['Spearman_rho']:.3f} (p = {corr['Spearman_p']:.2e})")
        print(f"  Pearson (signed wind) r = {corr['Pearson_signed_r']:.3f}")

    # --- 保存 CSV ---
    # 纬向统计
    zonal_merged = zonal_all['Holocene'][['lat']].copy()
    for period in ['Holocene', 'LGM']:
        z = zonal_all[period]
        zonal_merged[f'zonal_mean_D_{period}'] = z['zonal_mean_D'].values
        zonal_merged[f'zonal_std_D_{period}'] = z['zonal_std_D'].values
    zonal_merged['abs_wind'] = wind_all['Holocene']
    zonal_merged['signed_wind'] = interpolate_wind(df_wind, zonal_merged['lat'].values)
    zonal_csv = os.path.join(RESULTS, 'correlation_D_wind_zonal_stats.csv')
    zonal_merged.to_csv(zonal_csv, index=False)
    print(f'\n[SAVED] {zonal_csv}')

    # 相关系数汇总
    summary_csv = os.path.join(RESULTS, 'correlation_D_wind_summary.csv')
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f'[SAVED] {summary_csv}')

    # 区域统计
    df_regional = compute_regional_stats(grids, df_wind)
    regional_csv = os.path.join(RESULTS, 'correlation_D_wind_regional.csv')
    df_regional.to_csv(regional_csv, index=False)
    print(f'[SAVED] {regional_csv}')

    # --- 绘图 ---
    plot_zonal_profiles(zonal_all, wind_all, corr_all)
    plot_scatter_D_wind(zonal_all, wind_all, corr_all)
    plot_map_overlay(grids, df_wind)
    plot_regional_bars(df_regional)

    print('\n[DONE] C2 complete!')
