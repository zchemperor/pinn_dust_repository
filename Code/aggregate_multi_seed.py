import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 路径配置
RESULTS_PATH = "Data/model_results/"


def aggregate_training_points(pattern, output_file):
    files = sorted(glob.glob(os.path.join(RESULTS_PATH, pattern)))

    if not files:
        print(f"[WARNING] No files found for pattern: {pattern}")
        return None, None

    results = []
    for f in files:
        # 从文件名提取种子号
        basename = os.path.basename(f)
        try:
            seed = int(basename.split('seed')[1].split('_')[0].split('.')[0])
        except (IndexError, ValueError):
            print(f"[WARNING] Cannot extract seed from: {basename}")
            continue

        df = pd.read_csv(f)

        # 计算指标
        rmse = np.sqrt(mean_squared_error(df['log_dep'], df['PINN_log_dep']))
        mae = mean_absolute_error(df['log_dep'], df['PINN_log_dep'])
        r2 = r2_score(df['log_dep'], df['PINN_log_dep'])

        results.append({
            'seed': seed,
            'N': len(df),
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })

    if not results:
        print(f"[WARNING] No valid results for pattern: {pattern}")
        return None, None

    df_results = pd.DataFrame(results).sort_values('seed')

    # 计算汇总统计量
    summary = {
        'N_seeds': len(df_results),
        'seeds': df_results['seed'].tolist(),
        'R2_mean': df_results['R2'].mean(),
        'R2_std': df_results['R2'].std(),
        'R2_min': df_results['R2'].min(),
        'R2_max': df_results['R2'].max(),
        'RMSE_mean': df_results['RMSE'].mean(),
        'RMSE_std': df_results['RMSE'].std(),
        'MAE_mean': df_results['MAE'].mean(),
        'MAE_std': df_results['MAE'].std(),
    }

    # 打印结果
    print(f"\n{'='*60}")
    print(f"Pattern: {pattern}")
    print(f"Output: {output_file}")
    print(f"{'='*60}")
    print(f"Seeds: {summary['seeds']}")
    print(f"N samples: {df_results['N'].iloc[0]}")
    print(f"R2:   {summary['R2_mean']:.4f} +/- {summary['R2_std']:.4f} (range: {summary['R2_min']:.4f} - {summary['R2_max']:.4f})")
    print(f"RMSE: {summary['RMSE_mean']:.4f} +/- {summary['RMSE_std']:.4f}")
    print(f"MAE:  {summary['MAE_mean']:.4f} +/- {summary['MAE_std']:.4f}")

    # 保存结果
    output_path = os.path.join(RESULTS_PATH, output_file)
    df_results.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return df_results, summary


def aggregate_global_grid(pattern, output_file):
    files = sorted(glob.glob(os.path.join(RESULTS_PATH, pattern)))

    if not files:
        print(f"[WARNING] No files found for pattern: {pattern}")
        return None

    # 读取所有种子的结果
    dfs = []
    seeds = []
    for f in files:
        basename = os.path.basename(f)
        try:
            seed = int(basename.split('seed')[1].split('_')[0].split('.')[0])
        except (IndexError, ValueError):
            continue

        df = pd.read_csv(f)
        df['seed'] = seed
        dfs.append(df)
        seeds.append(seed)

    if not dfs:
        return None

    df_all = pd.concat(dfs, ignore_index=True)

    # 按位置分组计算统计量
    agg_dict = {
        'PINN_log_dep': ['mean', 'std', 'min', 'max']
    }

    # 如果有扩散系数列
    if 'PINN_Diffusivity' in df_all.columns:
        agg_dict['PINN_Diffusivity'] = ['mean', 'std', 'min', 'max']

    summary = df_all.groupby(['lon', 'lat']).agg(agg_dict).reset_index()

    # 展平列名
    summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                       for col in summary.columns]

    # 保存结果
    output_path = os.path.join(RESULTS_PATH, output_file)
    summary.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"Global Grid Aggregation")
    print(f"{'='*60}")
    print(f"Seeds: {seeds}")
    print(f"Grid points: {len(summary)}")
    print(f"Mean PINN_log_dep std across grid: {summary['PINN_log_dep_std'].mean():.4f}")
    if 'PINN_Diffusivity_std' in summary.columns:
        print(f"Mean PINN_Diffusivity std across grid: {summary['PINN_Diffusivity_std'].mean():.6f}")
    print(f"Saved to: {output_path}")

    return summary


def main():
    print("\n" + "="*70)
    print("Multi-Seed Experiment Aggregation")
    print("="*70)

    # 1. 汇总 v3 Emp-LGM 训练点
    aggregate_training_points(
        "df_pinn_empirical_LGM_training_points_seed*.csv",
        "df_pinn_empirical_LGM_seed_summary.csv"
    )

    # 2. 汇总 v3 Emp-Holocene 训练点
    aggregate_training_points(
        "df_pinn_empirical_Holocene_training_points_seed*.csv",
        "df_pinn_empirical_Holocene_seed_summary.csv"
    )

    # 3. 汇总 v3 Emp-LGM 全局网格
    aggregate_global_grid(
        "df_pinn_empirical_LGM_global_grid_seed*.csv",
        "df_pinn_empirical_LGM_global_grid_seed_summary.csv"
    )

    # 4. 汇总 v3 Emp-Holocene 全局网格
    aggregate_global_grid(
        "df_pinn_empirical_Holocene_global_grid_seed*.csv",
        "df_pinn_empirical_Holocene_global_grid_seed_summary.csv"
    )

    print("\n" + "="*70)
    print("Aggregation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

