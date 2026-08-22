"""Prepare auditable PMIP4 850 hPa 2D wind forcing on the project grid.

The script reads only Data/external_raw/*.nc and writes reproducible products
to Data/processed_data/pmip4_wind. It does not modify legacy v1/v2/v3 data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "Data" / "external_raw"
GRID_FILE = ROOT / "Data" / "processed_data" / "df_global_grid.csv"
OUT_DIR = ROOT / "Data" / "processed_data" / "pmip4_wind"
EARTH_RADIUS_M = 6_371_000.0
PRESSURE_LEVEL_PA = 85_000.0


class MissingForcingValuesError(ValueError):
    """Raised when a selected pressure-level field has an archive mask."""


@dataclass(frozen=True)
class ForcingSpec:
    forcing_id: str
    institution: str
    model: str
    member: str
    version: str
    start_year: int
    end_year: int
    role: str


FORCINGS = (
    ForcingSpec(
        "awi_primary_3981_4000",
        "AWI",
        "AWI-ESM-1-1-LR",
        "r1i1p1f1",
        "v20200212",
        3981,
        4000,
        "primary_forcing",
    ),
    ForcingSpec(
        "awi_early_3961_3980",
        "AWI",
        "AWI-ESM-1-1-LR",
        "r1i1p1f1",
        "v20200212",
        3961,
        3980,
        "temporal_robustness",
    ),
    ForcingSpec(
        "mpi_2330_2349",
        "MPI-M",
        "MPI-ESM1-2-LR",
        "r1i1p1f1",
        "v20190710",
        2330,
        2349,
        "model_sensitivity",
    ),
    ForcingSpec(
        "miroc_3280_3299",
        "MIROC",
        "MIROC-ES2L",
        "r1i1p1f2",
        "v20191002",
        3280,
        3299,
        "model_sensitivity",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_grid() -> tuple[np.ndarray, np.ndarray]:
    grid = pd.read_csv(GRID_FILE)
    lon = np.sort(grid["lon"].unique().astype(np.float64))
    lat = np.sort(grid["lat"].unique().astype(np.float64))
    if lon.size != 120 or lat.size != 60 or len(grid) != lon.size * lat.size:
        raise ValueError("df_global_grid.csv is not the expected 60 x 120 rectangular grid")
    if not (np.isclose(lon[0], -180.0) and np.isclose(lon[-1], 180.0)):
        raise ValueError("Target longitude grid must retain the -180/180 periodic seam")
    return lon, lat


def matching_files(spec: ForcingSpec, variable: str) -> list[Path]:
    pattern = f"{variable}_Amon_{spec.model}_lgm_{spec.member}_gn_*.nc"
    files = sorted(RAW_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {variable} files found for {spec.forcing_id}")
    return files


def read_component(spec: ForcingSpec, variable: str) -> tuple[np.ndarray, list[int], list[Path], np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    months: list[int] = []
    selected_files: list[Path] = []
    shared_lat: np.ndarray | None = None
    shared_lon: np.ndarray | None = None

    for path in matching_files(spec, variable):
        with Dataset(path) as dataset:
            if getattr(dataset, "source_id", None) != spec.model:
                raise ValueError(f"{path.name}: source_id mismatch")
            if getattr(dataset, "experiment_id", None) != "lgm":
                raise ValueError(f"{path.name}: experiment_id is not lgm")
            if getattr(dataset, "variant_label", None) != spec.member:
                raise ValueError(f"{path.name}: variant_label mismatch")
            if variable not in dataset.variables:
                raise ValueError(f"{path.name}: variable {variable} is missing")
            wind = dataset.variables[variable]
            if getattr(wind, "units", None) != "m s-1":
                raise ValueError(f"{path.name}: unexpected wind units {getattr(wind, 'units', None)!r}")
            plev = np.asarray(dataset.variables["plev"][:], dtype=np.float64)
            plev_index = np.flatnonzero(np.isclose(plev, PRESSURE_LEVEL_PA, atol=0.5))
            if plev_index.size != 1:
                raise ValueError(f"{path.name}: exact 850 hPa level is missing or duplicated")

            lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
            lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
            if shared_lat is None:
                shared_lat, shared_lon = lat, lon
            elif not (np.array_equal(shared_lat, lat) and np.array_equal(shared_lon, lon)):
                raise ValueError(f"{path.name}: grid differs within {spec.forcing_id}")

            time = dataset.variables["time"]
            dates = num2date(
                time[:],
                time.units,
                getattr(time, "calendar", "standard"),
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=False,
            )
            selected = [index for index, date in enumerate(dates) if spec.start_year <= date.year <= spec.end_year]
            if not selected:
                continue
            values = np.ma.filled(wind[selected, int(plev_index[0]), :, :], np.nan).astype(np.float64)
            chunks.append(values)
            months.extend(int(dates[index].month) for index in selected)
            selected_files.append(path)

    if not chunks or shared_lat is None or shared_lon is None:
        raise ValueError(f"{spec.forcing_id}: no records selected for {variable}")
    values = np.concatenate(chunks, axis=0)
    expected_months = 12 * (spec.end_year - spec.start_year + 1)
    if values.shape[0] != expected_months:
        raise ValueError(f"{spec.forcing_id}: expected {expected_months} months, found {values.shape[0]}")
    if sorted(months.count(month) for month in range(1, 13)) != [expected_months // 12] * 12:
        raise ValueError(f"{spec.forcing_id}: incomplete monthly climatology for {variable}")
    if not np.isfinite(values).all():
        raise MissingForcingValuesError(f"{spec.forcing_id}: {variable} contains missing values at 850 hPa")
    return values, months, selected_files, shared_lat, shared_lon


def monthly_climatology(values: np.ndarray, months: list[int]) -> np.ndarray:
    return np.stack([values[np.asarray(months) == month].mean(axis=0) for month in range(1, 13)], axis=0)


def regrid_periodic(field: np.ndarray, source_lat: np.ndarray, source_lon: np.ndarray, target_lat: np.ndarray, target_lon: np.ndarray) -> tuple[np.ndarray, list[float]]:
    if not (np.all(np.diff(source_lat) > 0) and np.all(np.diff(source_lon) > 0)):
        raise ValueError("Source coordinates must be strictly increasing")
    if not (np.all(np.diff(target_lat) > 0) and np.all(np.diff(target_lon) > 0)):
        raise ValueError("Target coordinates must be strictly increasing")

    source_lon = np.mod(source_lon, 360.0)
    order = np.argsort(source_lon)
    source_lon = source_lon[order]
    field = field[..., order]
    if np.any(np.diff(source_lon) <= 0):
        raise ValueError("Source longitude is not unique after normalization")

    periodic_lon = np.concatenate([source_lon, [source_lon[0] + 360.0]])
    periodic_field = np.concatenate([field, field[..., :1]], axis=-1)
    clipped_lat = np.clip(target_lat, source_lat[0], source_lat[-1])
    polar_clamped = target_lat[(target_lat < source_lat[0]) | (target_lat > source_lat[-1])].tolist()
    lon_query = np.mod(target_lon, 360.0)
    lat_mesh, lon_mesh = np.meshgrid(clipped_lat, lon_query, indexing="ij")
    points = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])

    flat = periodic_field.reshape((-1, periodic_field.shape[-2], periodic_field.shape[-1]))
    out = np.empty((flat.shape[0], target_lat.size, target_lon.size), dtype=np.float64)
    for index, slice_ in enumerate(flat):
        interpolator = RegularGridInterpolator((source_lat, periodic_lon), slice_, method="linear", bounds_error=True)
        out[index] = interpolator(points).reshape(target_lat.size, target_lon.size)
    out = out.reshape(field.shape[:-2] + (target_lat.size, target_lon.size))
    if not np.allclose(out[..., 0], out[..., -1], rtol=0.0, atol=1e-12):
        raise AssertionError("Periodic target seam was not preserved")
    return out, polar_clamped


def write_netcdf(path: Path, spec: ForcingSpec, lon: np.ndarray, lat: np.ndarray, ua_monthly: np.ndarray, va_monthly: np.ndarray, polar_clamped: list[float]) -> None:
    annual_ua = ua_monthly.mean(axis=0)
    annual_va = va_monthly.mean(axis=0)
    with Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("month", 12)
        dataset.createDimension("lat", lat.size)
        dataset.createDimension("lon", lon.size)
        dataset.createDimension("polar_clamped_lat", len(polar_clamped))
        dataset.title = "PMIP4 LGM 850 hPa horizontal wind forcing on the PINN project grid"
        dataset.forcing_id = spec.forcing_id
        dataset.source_id = spec.model
        dataset.institution_id = spec.institution
        dataset.variant_label = spec.member
        dataset.version = spec.version
        dataset.experiment_id = "lgm"
        dataset.role = spec.role
        dataset.time_window = f"{spec.start_year}-01 through {spec.end_year}-12"
        dataset.temporal_aggregation = "monthly climatology and unweighted annual mean of monthly climatology"
        dataset.pressure_level_pa = PRESSURE_LEVEL_PA
        dataset.target_grid_source = str(GRID_FILE.relative_to(ROOT)).replace("\\", "/")
        dataset.polar_boundary_treatment = "Only target latitudes outside the source grid are clipped to the nearest source latitude; these rows are outside the legacy PINN active domain."
        dataset.created_utc = datetime.now(timezone.utc).isoformat()
        dataset.earth_radius_m = EARTH_RADIUS_M

        for name, values, units in (("lon", lon, "degrees_east"), ("lat", lat, "degrees_north")):
            variable = dataset.createVariable(name, "f8", (name,))
            variable[:] = values
            variable.units = units
        month = dataset.createVariable("month", "i4", ("month",))
        month[:] = np.arange(1, 13)
        month.long_name = "calendar_month"
        clamped = dataset.createVariable("polar_clamped_lat", "f8", ("polar_clamped_lat",))
        clamped[:] = np.asarray(polar_clamped, dtype=np.float64)
        clamped.units = "degrees_north"

        for name, values, dimensions, long_name in (
            ("ua_monthly", ua_monthly, ("month", "lat", "lon"), "monthly climatological eastward wind"),
            ("va_monthly", va_monthly, ("month", "lat", "lon"), "monthly climatological northward wind"),
            ("ua_annual", annual_ua, ("lat", "lon"), "annual climatological eastward wind"),
            ("va_annual", annual_va, ("lat", "lon"), "annual climatological northward wind"),
            ("wind_speed_annual", np.hypot(annual_ua, annual_va), ("lat", "lon"), "annual climatological horizontal wind speed"),
        ):
            variable = dataset.createVariable(name, "f8", dimensions, zlib=True, complevel=4)
            variable[:] = values
            variable.units = "m s-1"
            variable.long_name = long_name


def process_forcing(spec: ForcingSpec, lon: np.ndarray, lat: np.ndarray) -> tuple[dict[str, object], list[dict[str, object]]]:
    ua_values, ua_months, ua_files, source_lat, source_lon = read_component(spec, "ua")
    va_values, va_months, va_files, va_lat, va_lon = read_component(spec, "va")
    if not (np.array_equal(source_lat, va_lat) and np.array_equal(source_lon, va_lon)):
        raise ValueError(f"{spec.forcing_id}: ua/va source grids differ")
    if ua_months != va_months:
        raise ValueError(f"{spec.forcing_id}: ua/va time selections differ")

    ua_monthly_native = monthly_climatology(ua_values, ua_months)
    va_monthly_native = monthly_climatology(va_values, va_months)
    ua_monthly, polar_clamped = regrid_periodic(ua_monthly_native, source_lat, source_lon, lat, lon)
    va_monthly, va_polar_clamped = regrid_periodic(va_monthly_native, source_lat, source_lon, lat, lon)
    if polar_clamped != va_polar_clamped:
        raise AssertionError("ua and va use different polar boundary treatment")

    output_nc = OUT_DIR / f"{spec.forcing_id}_annual_850hpa.nc"
    output_npz = OUT_DIR / f"{spec.forcing_id}_annual_850hpa.npz"
    write_netcdf(output_nc, spec, lon, lat, ua_monthly, va_monthly, polar_clamped)
    np.savez_compressed(
        output_npz,
        lon=lon,
        lat=lat,
        month=np.arange(1, 13),
        ua_monthly=ua_monthly,
        va_monthly=va_monthly,
        ua_annual=ua_monthly.mean(axis=0),
        va_annual=va_monthly.mean(axis=0),
        wind_speed_annual=np.hypot(ua_monthly.mean(axis=0), va_monthly.mean(axis=0)),
        pressure_level_pa=np.asarray(PRESSURE_LEVEL_PA),
        earth_radius_m=np.asarray(EARTH_RADIUS_M),
    )

    input_files = sorted(set(ua_files + va_files))
    metadata = {
        **asdict(spec),
        "pressure_level_pa": PRESSURE_LEVEL_PA,
        "monthly_samples_per_component": int(ua_values.shape[0]),
        "source_grid": {"lat": int(source_lat.size), "lon": int(source_lon.size)},
        "target_grid": {"lat": int(lat.size), "lon": int(lon.size)},
        "polar_clamped_latitudes": polar_clamped,
        "input_files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in input_files],
        "input_sha256": {path.name: sha256(path) for path in input_files},
        "outputs": {
            "netcdf": str(output_nc.relative_to(ROOT)).replace("\\", "/"),
            "netcdf_sha256": sha256(output_nc),
            "npz": str(output_npz.relative_to(ROOT)).replace("\\", "/"),
            "npz_sha256": sha256(output_npz),
        },
    }
    qc_rows: list[dict[str, object]] = []
    for component, values in (("ua", ua_monthly.mean(axis=0)), ("va", va_monthly.mean(axis=0)), ("wind_speed", np.hypot(ua_monthly.mean(axis=0), va_monthly.mean(axis=0)))):
        qc_rows.append(
            {
                "forcing_id": spec.forcing_id,
                "component": component,
                "finite_fraction": float(np.isfinite(values).mean()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "seam_max_abs_difference": float(np.max(np.abs(values[:, 0] - values[:, -1]))),
                "polar_clamped_rows": len(polar_clamped),
            }
        )
    return metadata, qc_rows


def main() -> None:
    if not RAW_DIR.exists() or len(list(RAW_DIR.glob("*.nc"))) != 20:
        raise RuntimeError("Data/external_raw must contain exactly the 20 validated PMIP4 NetCDF files")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lon, lat = target_grid()
    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "raw_directory": str(RAW_DIR.relative_to(ROOT)).replace("\\", "/"),
        "target_grid": {"lat": int(lat.size), "lon": int(lon.size)},
        "pressure_level_pa": PRESSURE_LEVEL_PA,
        "status": "complete",
        "forcings": [],
        "ineligible_forcings": [],
    }
    rows: list[dict[str, object]] = []
    for spec in FORCINGS:
        print(f"Preparing {spec.forcing_id}", flush=True)
        try:
            metadata, qc_rows = process_forcing(spec, lon, lat)
        except MissingForcingValuesError as error:
            if spec.model != "MIROC-ES2L":
                raise
            manifest["status"] = "partial_miroc_850hpa_mask_excluded"
            manifest["ineligible_forcings"].append(
                {
                    **asdict(spec),
                    "reason": str(error),
                    "decision": "Excluded from formal forcing sensitivity. No gap-filled field is produced.",
                }
            )
            print(f"Excluded {spec.forcing_id}: {error}", flush=True)
            continue
        manifest["forcings"].append(metadata)
        rows.extend(qc_rows)
    (OUT_DIR / "pmip4_forcing_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT_DIR / "pmip4_preprocessing_qc.csv", index=False)
    print(f"Wrote {len(manifest['forcings'])} eligible forcing products to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
