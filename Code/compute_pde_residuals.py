import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from scipy.interpolate import interp1d
from style_jcp import apply_style, SAVE_DPI, save_figure_all_formats

apply_style()


BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..')
DATA = os.path.join(ROOT, 'Data')
RESULTS = os.path.join(DATA, 'model_results')
FIGURES = os.path.join(ROOT, 'Figures')
ABL = os.path.join(ROOT, 'Ablation_Result')


def read_D_from_variables(path):
    with open(path) as f:
        last = f.readlines()[-1].strip()
    nums = re.findall(r'[-+]?\d*\.?\d+[eE][-+]?\d+|[-+]?\d*\.?\d+', last)
    return float(nums[1])  # [step, D, north, south]

def load_wind(data_path):
    df_w = pd.read_csv(os.path.join(data_path, 'processed_data', 'df_wind.csv'))
    lat_norm = df_w['latitude'].values / 90.0
    wind_norm = df_w['wind'].values / df_w['wind'].max()
    return interp1d(lat_norm, wind_norm, kind='linear', fill_value='extrapolate')

def reshape_to_grid(df, col, lats, lons):
    nlat, nlon = len(lats), len(lons)
    grid = np.full((nlat, nlon), np.nan)
    for _, row in df.iterrows():
        i = np.argmin(np.abs(lats - row['lat']))
        j = np.argmin(np.abs(lons - row['lon']))
        grid[i, j] = row[col]
    return grid


def compute_pde_residual_constD(u_grid, D_scalar, lats_rad, lons_rad, K_vec):
    nlat, nlon = u_grid.shape
    dlam = lons_rad[1] - lons_rad[0]
    dtheta = lats_rad[1] - lats_rad[0]

    du_dlam = np.zeros_like(u_grid)
    du_dlam[:, 1:-1] = (u_grid[:, 2:] - u_grid[:, :-2]) / (2 * dlam)
    du_dlam[:, 0] = (u_grid[:, 1] - u_grid[:, -2]) / (2 * dlam)
    du_dlam[:, -1] = (u_grid[:, 1] - u_grid[:, -2]) / (2 * dlam)

    d2u_dlam2 = np.zeros_like(u_grid)
    d2u_dlam2[:, 1:-1] = (u_grid[:, 2:] - 2*u_grid[:, 1:-1] + u_grid[:, :-2]) / (dlam**2)
    d2u_dlam2[:, 0] = (u_grid[:, 1] - 2*u_grid[:, 0] + u_grid[:, -2]) / (dlam**2)
    d2u_dlam2[:, -1] = (u_grid[:, 1] - 2*u_grid[:, -1] + u_grid[:, -2]) / (dlam**2)

    du_dtheta = np.zeros_like(u_grid)
    du_dtheta[1:-1, :] = (u_grid[2:, :] - u_grid[:-2, :]) / (2 * dtheta)

    d2u_dtheta2 = np.zeros_like(u_grid)
    d2u_dtheta2[1:-1, :] = (u_grid[2:, :] - 2*u_grid[1:-1, :] + u_grid[:-2, :]) / (dtheta**2)

    cos_theta = np.cos(lats_rad)[:, None]
    tan_theta = np.tan(lats_rad)[:, None]
    K = K_vec[:, None]

    advection = -K / cos_theta * du_dlam
    diffusion = D_scalar * (1.0 / cos_theta**2 * d2u_dlam2 + d2u_dtheta2 - tan_theta * du_dtheta)
    residual = advection + diffusion

    return residual[1:-1, :]


def compute_pde_residual_Dx(u_grid, D_grid, lats_rad, lons_rad, K_vec):
    nlat, nlon = u_grid.shape
    dlam = lons_rad[1] - lons_rad[0]
    dtheta = lats_rad[1] - lats_rad[0]

    du_dlam = np.zeros_like(u_grid)
    du_dlam[:, 1:-1] = (u_grid[:, 2:] - u_grid[:, :-2]) / (2 * dlam)
    du_dlam[:, 0] = (u_grid[:, 1] - u_grid[:, -2]) / (2 * dlam)
    du_dlam[:, -1] = (u_grid[:, 1] - u_grid[:, -2]) / (2 * dlam)

    d2u_dlam2 = np.zeros_like(u_grid)
    d2u_dlam2[:, 1:-1] = (u_grid[:, 2:] - 2*u_grid[:, 1:-1] + u_grid[:, :-2]) / (dlam**2)
    d2u_dlam2[:, 0] = (u_grid[:, 1] - 2*u_grid[:, 0] + u_grid[:, -2]) / (dlam**2)
    d2u_dlam2[:, -1] = (u_grid[:, 1] - 2*u_grid[:, -1] + u_grid[:, -2]) / (dlam**2)

    du_dtheta = np.zeros_like(u_grid)
    du_dtheta[1:-1, :] = (u_grid[2:, :] - u_grid[:-2, :]) / (2 * dtheta)

    d2u_dtheta2 = np.zeros_like(u_grid)
    d2u_dtheta2[1:-1, :] = (u_grid[2:, :] - 2*u_grid[1:-1, :] + u_grid[:-2, :]) / (dtheta**2)

    dD_dlam = np.zeros_like(D_grid)
    dD_dlam[:, 1:-1] = (D_grid[:, 2:] - D_grid[:, :-2]) / (2 * dlam)
    dD_dlam[:, 0] = (D_grid[:, 1] - D_grid[:, -2]) / (2 * dlam)
    dD_dlam[:, -1] = (D_grid[:, 1] - D_grid[:, -2]) / (2 * dlam)

    dD_dtheta = np.zeros_like(D_grid)
    dD_dtheta[1:-1, :] = (D_grid[2:, :] - D_grid[:-2, :]) / (2 * dtheta)

    cos_theta = np.cos(lats_rad)[:, None]
    tan_theta = np.tan(lats_rad)[:, None]
    K = K_vec[:, None]

    advection = -K / cos_theta * du_dlam
    diffusion_main = D_grid * (1.0/cos_theta**2 * d2u_dlam2 + d2u_dtheta2 - tan_theta * du_dtheta)
    diffusion_grad = 1.0/cos_theta**2 * dD_dlam * du_dlam + dD_dtheta * du_dtheta
    residual = advection + diffusion_main + diffusion_grad

    return residual[1:-1, :]


def compute_bc_violation(u_grid, lats_deg):
    periodic_diff = np.abs(u_grid[:, 0] - u_grid[:, -1])
    periodic_mean = np.mean(periodic_diff)
    periodic_max = np.max(periodic_diff)

    polar_north_std = np.std(u_grid[-2, :])
    polar_south_std = np.std(u_grid[1, :])

    return periodic_mean, periodic_max, polar_north_std, polar_south_std


MODELS = {
    'v0': {
        'grid_path': lambda p: os.path.join(ABL, 'inn0', 'Data', 'model_results', f'df_pinn_empirical_{p}_global_grid.csv'),
        'var_path': lambda p: os.path.join(ABL, 'inn0', 'Data', 'trained_models', f'model_empirical_{p}', 'variables.dat'),
        'has_Dx': False, 'periods': ['Holocene', 'LGM'],
    },
    'v1': {
        'grid_path': lambda p: os.path.join(ABL, 'inn1', 'Data', 'model_results', f'df_pinn_empirical_{p}_global_grid.csv'),
        'var_path': lambda p: os.path.join(ABL, 'inn1', 'Data', 'trained_models', f'model_empirical_{p}', 'variables.dat'),
        'has_Dx': False, 'periods': ['Holocene', 'LGM'],
    },
    'v2': {
        'grid_path': lambda p: os.path.join(ABL, 'inn2', 'Data', 'model_results', f'df_pinn_empirical_{p}_global_grid.csv'),
        'var_path': lambda p: os.path.join(ABL, 'inn2', 'Data', 'trained_models', f'model_empirical_{p}', 'variables.dat'),
        'has_Dx': False, 'periods': ['Holocene', 'LGM'],
    },
    'v3': {
        'grid_path': lambda p: os.path.join(RESULTS, f'df_pinn_empirical_{p}_global_grid.csv'),
        'var_path': None,
        'has_Dx': True, 'periods': ['Holocene', 'LGM'],
    },
    'constD': {
        'grid_path': lambda p: os.path.join(RESULTS, f'df_pinn_empirical_{p}_constD_global_grid.csv'),
        'var_path': lambda p: os.path.join(DATA, 'trained_models', f'model_empirical_{p}_constD', 'variables.dat'),
        'has_Dx': False, 'periods': ['LGM'],  # 只有 LGM
    },
}


if __name__ == '__main__':
    # 加载全球网格坐标
    df_grid = pd.read_csv(os.path.join(DATA, 'processed_data', 'df_global_grid.csv'))
    lons_deg = np.sort(df_grid['lon'].unique())
    lats_deg = np.sort(df_grid['lat'].unique())
    lons_rad = np.deg2rad(lons_deg)
    lats_rad = np.deg2rad(lats_deg)

    # 加载风场
    wind_interp = load_wind(DATA)
    K_vec = wind_interp(lats_deg / 90.0).astype(np.float64)

    results = []

    for model_name, cfg in MODELS.items():
        for period in cfg['periods']:
            grid_path = cfg['grid_path'](period)
            if not os.path.exists(grid_path):
                print(f"[SKIP] {model_name} {period}: file not found {grid_path}")
                continue

            df = pd.read_csv(grid_path)
            u_grid = reshape_to_grid(df, 'PINN_log_dep', lats_deg, lons_deg)

            if cfg['has_Dx']:
                D_grid = reshape_to_grid(df, 'PINN_Diffusivity', lats_deg, lons_deg)
                res = compute_pde_residual_Dx(u_grid, D_grid, lats_rad, lons_rad, K_vec)
            else:
                D_val = read_D_from_variables(cfg['var_path'](period))
                res = compute_pde_residual_constD(u_grid, D_val, lats_rad, lons_rad, K_vec)

            # BC violation
            p_mean, p_max, pn_std, ps_std = compute_bc_violation(u_grid, lats_deg)

            # 统计量
            abs_res = np.abs(res)
            row = {
                'Config': model_name, 'Period': period,
                'PDE_res_mean': np.mean(abs_res),
                'PDE_res_median': np.median(abs_res),
                'PDE_res_95pct': np.percentile(abs_res, 95),
                'PDE_res_max': np.max(abs_res),
                'BC_periodic_mean': p_mean,
                'BC_periodic_max': p_max,
                'BC_polar_north_std': pn_std,
                'BC_polar_south_std': ps_std,
            }
            results.append(row)
            print(f"[OK] {model_name:8s} {period:10s} | PDE mean={row['PDE_res_mean']:.4e} median={row['PDE_res_median']:.4e} 95%={row['PDE_res_95pct']:.4e} | BC periodic={p_mean:.4e} polar_N={pn_std:.4e} polar_S={ps_std:.4e}")

    # 保存 CSV
    df_out = pd.DataFrame(results)
    out_path = os.path.join(RESULTS, 'pde_residual_evaluation.csv')
    df_out.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}")

    for period in ['Holocene', 'LGM']:
        grid_path = MODELS['v3']['grid_path'](period)
        df = pd.read_csv(grid_path)
        u_grid = reshape_to_grid(df, 'PINN_log_dep', lats_deg, lons_deg)
        D_grid = reshape_to_grid(df, 'PINN_Diffusivity', lats_deg, lons_deg)
        res = compute_pde_residual_Dx(u_grid, D_grid, lats_rad, lons_rad, K_vec)

        fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.4))
        inner_lats = lats_deg[1:-1]
        im = ax.pcolormesh(lons_deg, inner_lats, np.log10(np.abs(res) + 1e-10),
                           cmap='RdYlBu_r', vmin=-5, vmax=0, shading='auto')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_title(f'v3 PDE Residual (log₁₀|R|) — Empirical {period}')
        plt.colorbar(im, ax=ax, label='log₁₀|PDE residual|')
        fig.tight_layout()
        fig_path = os.path.join(FIGURES, f'PDE_RESIDUAL_v3_EMPIRICAL_{period.upper()}')
        save_figure_all_formats(fig_path, fig=fig, dpi=SAVE_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"[FIG] {fig_path}.pdf/.png/.eps")

    print("\n[DONE] B1 complete!")
