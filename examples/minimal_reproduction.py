from __future__ import annotations
from pathlib import Path
import importlib.util
import sys
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "Code"
OUTPUT = Path(__file__).resolve().parent / "output"


REQUIRED_FILES = {
    "holocene_grid": ROOT / "Data" / "model_results" / "df_pinn_empirical_Holocene_global_grid.csv",
    "lgm_grid": ROOT / "Data" / "model_results" / "df_pinn_empirical_LGM_global_grid.csv",
    "wind": ROOT / "Data" / "processed_data" / "df_wind.csv",
}


def check_required_files() -> None:
    missing = [path for path in REQUIRED_FILES.values() if not path.exists()]
    if missing:
        msg = "\n".join(str(path.relative_to(ROOT)) for path in missing)
        raise FileNotFoundError(f"Missing required input files:\n{msg}")


def import_original_module():
    """Import Code/correlation_D_wind.py without running its main block."""
    sys.path.insert(0, str(CODE_DIR))
    module_path = CODE_DIR / "correlation_D_wind.py"
    spec = importlib.util.spec_from_file_location("correlation_D_wind_original", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_original_fig6_inputs(corr_module):
    grids, df_wind = corr_module.load_data()

    zonal_all = {}
    wind_all = {}
    corr_all = {}
    summary_rows = []

    for period in ["Holocene", "LGM"]:
        zonal = corr_module.compute_zonal_stats(grids[period])
        zonal = zonal[(zonal["lat"] > -89) & (zonal["lat"] < 89)].copy()

        lats = zonal["lat"].values
        wind_interp = corr_module.interpolate_wind(df_wind, lats)
        abs_wind = abs(wind_interp)
        corr = corr_module.compute_correlations(
            zonal["zonal_mean_D"].values,
            abs_wind,
            wind_interp,
        )

        zonal_all[period] = zonal
        wind_all[period] = abs_wind
        corr_all[period] = corr

        summary_rows.append(
            {
                "Period": period,
                "Pearson_r": corr["Pearson_r"],
                "Pearson_p": corr["Pearson_p"],
                "Spearman_rho": corr["Spearman_rho"],
                "Spearman_p": corr["Spearman_p"],
                "Pearson_signed_r": corr["Pearson_signed_r"],
                "Pearson_signed_p": corr["Pearson_signed_p"],
            }
        )

    return zonal_all, wind_all, corr_all, pd.DataFrame(summary_rows)


def main() -> None:
    check_required_files()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    corr_module = import_original_module()

    corr_module.FIGURES = str(OUTPUT)

    zonal_all, wind_all, corr_all, summary = compute_original_fig6_inputs(corr_module)

    corr_module.plot_zonal_profiles(zonal_all, wind_all, corr_all)
    corr_module.plot_scatter_D_wind(zonal_all, wind_all, corr_all)

    summary.to_csv(OUTPUT / "minimal_reproduction_summary.csv", index=False)

    print("Regenerated manuscript Fig. 6 source panels with original plotting code.")
    print(f"Output directory: {OUTPUT.relative_to(ROOT)}")
    print("\nSummary statistics:")
    print(summary.to_string(index=False))
    print("\nGenerated files:")
    for name in [
        "CORRELATION_ZONAL_PROFILE_D_WIND.png",
        "CORRELATION_ZONAL_PROFILE_D_WIND.pdf",
        "CORRELATION_SCATTER_D_WIND.png",
        "CORRELATION_SCATTER_D_WIND.pdf",
        "minimal_reproduction_summary.csv",
    ]:
        path = OUTPUT / name
        print(f"  - {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
