"""Create PNG and EPS QC figures for prepared PMIP4 2D wind forcing."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset

from prepare_pmip4_2d_wind import FORCINGS, OUT_DIR, ROOT
from style_jcp import apply_style, save_figure_all_formats


FIGURES = ROOT / "Figures"


def load(forcing_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(OUT_DIR / f"{forcing_id}_annual_850hpa.nc") as dataset:
        return (
            np.asarray(dataset.variables["lon"][:]),
            np.asarray(dataset.variables["lat"][:]),
            np.asarray(dataset.variables["ua_annual"][:]),
            np.asarray(dataset.variables["va_annual"][:]),
        )


def save_pair(figure: plt.Figure, stem: str) -> None:
    save_figure_all_formats(FIGURES / stem, figure)
    plt.close(figure)


def plot_forcing(spec) -> None:
    lon, lat, ua, va = load(spec.forcing_id)
    speed = np.hypot(ua, va)
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fields = ((ua, "Eastward wind U (m s-1)", "RdBu_r"), (va, "Northward wind V (m s-1)", "RdBu_r"), (speed, "Wind speed (m s-1)", "viridis"))
    for axis, (field, title, cmap) in zip(axes.flat[:3], fields):
        limit = np.nanmax(np.abs(field)) if cmap == "RdBu_r" else None
        image = axis.pcolormesh(lon, lat, field, shading="auto", cmap=cmap, vmin=-limit if limit else None, vmax=limit)
        axis.set_title(title)
        axis.set_xlabel("Longitude (degrees)")
        axis.set_ylabel("Latitude (degrees)")
        figure.colorbar(image, ax=axis, shrink=0.82)
    axis = axes.flat[3]
    axis.pcolormesh(lon, lat, speed, shading="auto", cmap="Greys")
    stride_lat, stride_lon = 4, 6
    axis.quiver(
        lon_mesh[::stride_lat, ::stride_lon],
        lat_mesh[::stride_lat, ::stride_lon],
        ua[::stride_lat, ::stride_lon],
        va[::stride_lat, ::stride_lon],
        scale=260,
        width=0.0022,
    )
    axis.set_title("Annual vector wind")
    axis.set_xlabel("Longitude (degrees)")
    axis.set_ylabel("Latitude (degrees)")
    figure.suptitle(f"PMIP4 LGM 850 hPa forcing: {spec.forcing_id}")
    save_pair(figure, f"PMIP4_{spec.forcing_id.upper()}_WIND_VECTOR")


def plot_comparison() -> None:
    reference_spec = FORCINGS[0]
    lon, lat, ref_u, ref_v = load(reference_spec.forcing_id)
    ref_speed = np.hypot(ref_u, ref_v)
    comparison_specs = [spec for spec in FORCINGS[1:] if (OUT_DIR / f"{spec.forcing_id}_annual_850hpa.nc").exists()]
    products = []
    for spec in comparison_specs:
        _, _, ua, va = load(spec.forcing_id)
        speed = np.hypot(ua, va)
        products.append((spec, speed - ref_speed, np.hypot(ua - ref_u, va - ref_v), ua, va))

    speed_limit = max(float(np.nanmax(np.abs(speed_delta))) for _, speed_delta, _, _, _ in products)
    vector_limit = max(float(np.nanmax(vector_delta)) for _, _, vector_delta, _, _ in products)
    figure, axes = plt.subplots(2, len(comparison_specs), figsize=(7.0, 5.35), constrained_layout=True)
    rows: list[dict[str, object]] = []
    top_images, bottom_images = [], []
    titles = ["AWI early minus AWI primary", "MPI minus AWI primary"]
    for column, ((spec, speed_delta, vector_delta, ua, va), title) in enumerate(zip(products, titles)):
        top_images.append(
            axes[0, column].pcolormesh(
                lon, lat, speed_delta, shading="auto", cmap="RdBu_r", vmin=-speed_limit, vmax=speed_limit
            )
        )
        bottom_images.append(
            axes[1, column].pcolormesh(
                lon, lat, vector_delta, shading="auto", cmap="magma", vmin=0.0, vmax=vector_limit
            )
        )
        axes[0, column].set_title(f"({chr(97 + column)}) {title}")
        axes[1, column].set_title(f"({chr(99 + column)}) Vector-difference magnitude")
        axes[1, column].set_xlabel("Longitude (degrees)")
        axes[0, column].set_ylabel("Latitude (degrees)")
        axes[1, column].set_ylabel("Latitude (degrees)")
        rows.append(
            {
                "forcing_id": spec.forcing_id,
                "reference": reference_spec.forcing_id,
                "u_spatial_correlation": float(np.corrcoef(ref_u.ravel(), ua.ravel())[0, 1]),
                "v_spatial_correlation": float(np.corrcoef(ref_v.ravel(), va.ravel())[0, 1]),
                "speed_spatial_correlation": float(np.corrcoef(ref_speed.ravel(), speed.ravel())[0, 1]),
                "mean_vector_difference_m_s": float(vector_delta.mean()),
            }
        )
    figure.colorbar(top_images[-1], ax=axes[0, :], pad=0.02, label="Wind-speed difference (m s$^{-1}$)")
    figure.colorbar(bottom_images[-1], ax=axes[1, :], pad=0.02, label="Vector-difference magnitude (m s$^{-1}$)")
    save_pair(figure, "PMIP4_FORCING_COMPARISON")
    pd.DataFrame(rows).to_csv(OUT_DIR / "pmip4_forcing_comparison.csv", index=False)


def main() -> None:
    apply_style()
    for spec in FORCINGS:
        if (OUT_DIR / f"{spec.forcing_id}_annual_850hpa.nc").exists():
            plot_forcing(spec)
    plot_comparison()
    print("Wrote PMIP4 forcing QC figures in PNG and EPS formats.")


if __name__ == "__main__":
    main()
