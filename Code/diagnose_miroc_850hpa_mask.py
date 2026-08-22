"""Document the MIROC-ES2L 850 hPa archive mask without gap filling it."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

from prepare_pmip4_2d_wind import FORCINGS, PRESSURE_LEVEL_PA, RAW_DIR, ROOT
from style_jcp import apply_style, save_figure_all_formats


OUT_DIR = ROOT / "Data" / "processed_data" / "pmip4_wind"
FIGURES = ROOT / "Figures"


def selected_mask(variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = next(item for item in FORCINGS if item.model == "MIROC-ES2L")
    path = next(RAW_DIR.glob(f"{variable}_Amon_{spec.model}_lgm_{spec.member}_gn_*.nc"))
    with Dataset(path) as dataset:
        plev = np.asarray(dataset.variables["plev"][:], dtype=np.float64)
        plev_index = int(np.flatnonzero(np.isclose(plev, PRESSURE_LEVEL_PA, atol=0.5))[0])
        time = dataset.variables["time"]
        dates = num2date(time[:], time.units, getattr(time, "calendar", "standard"))
        indices = [index for index, date in enumerate(dates) if 3280 <= date.year <= 3299]
        values = np.ma.asarray(dataset.variables[variable][indices, plev_index, :, :])
        return np.asarray(dataset.variables["lon"][:]), np.asarray(dataset.variables["lat"][:]), np.ma.getmaskarray(values)


def save_pair(figure: plt.Figure, stem: str) -> None:
    save_figure_all_formats(FIGURES / stem, figure)
    plt.close(figure)


def main() -> None:
    apply_style()
    lon, lat, ua_mask = selected_mask("ua")
    _, _, va_mask = selected_mask("va")
    if not np.array_equal(ua_mask, va_mask):
        raise AssertionError("MIROC ua and va masks are not identical")
    persistent = ua_mask.all(axis=0)
    frequency = ua_mask.mean(axis=0)
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    image = axes[0].pcolormesh(lon, lat, frequency, shading="auto", cmap="magma", vmin=0, vmax=1)
    axes[0].set_title("(a) Missing fraction, 3280-3299")
    figure.colorbar(image, ax=axes[0], label="Fraction of 240 monthly fields")
    image = axes[1].pcolormesh(lon, lat, persistent.astype(float), shading="auto", cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("(b) Cells unavailable in every month")
    figure.colorbar(image, ax=axes[1], label="Persistent unavailability (0/1)")
    for axis in axes:
        axis.set_xlabel("Longitude (degrees)")
        axis.set_ylabel("Latitude (degrees)")
    save_pair(figure, "PMIP4_MIROC_850HPA_MASK_DIAGNOSTIC")
    summary = pd.DataFrame(
        [
            {
                "forcing_id": "miroc_3280_3299",
                "component": "ua_va_shared_mask",
                "months": int(ua_mask.shape[0]),
                "missing_fraction": float(ua_mask.mean()),
                "cells_missing_any_month": int(ua_mask.any(axis=0).sum()),
                "cells_persistently_missing": int(persistent.sum()),
                "decision": "excluded_from_formal_850hpa_forcing_sensitivity_without_gap_filling",
            }
        ]
    )
    summary.to_csv(OUT_DIR / "miroc_850hpa_mask_qc.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
