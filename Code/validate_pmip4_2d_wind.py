"""Validate prepared PMIP4 2D wind forcing products before PINN training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from prepare_pmip4_2d_wind import FORCINGS, OUT_DIR, PRESSURE_LEVEL_PA


def validate(path: Path) -> None:
    with Dataset(path) as dataset:
        if dataset.dimensions["lat"].size != 60 or dataset.dimensions["lon"].size != 120:
            raise AssertionError(f"{path.name}: target grid is not 60 x 120")
        if dataset.dimensions["month"].size != 12:
            raise AssertionError(f"{path.name}: monthly climatology is incomplete")
        if not np.isclose(float(dataset.pressure_level_pa), PRESSURE_LEVEL_PA):
            raise AssertionError(f"{path.name}: pressure level metadata mismatch")
        lon = np.asarray(dataset.variables["lon"][:])
        if not (np.isclose(lon[0], -180.0) and np.isclose(lon[-1], 180.0)):
            raise AssertionError(f"{path.name}: periodic longitude seam is missing")
        for name in ("ua_monthly", "va_monthly", "ua_annual", "va_annual", "wind_speed_annual"):
            values = np.asarray(dataset.variables[name][:], dtype=np.float64)
            if not np.isfinite(values).all():
                raise AssertionError(f"{path.name}: {name} contains non-finite values")
            if not np.allclose(values[..., 0], values[..., -1], rtol=0.0, atol=1e-12):
                raise AssertionError(f"{path.name}: {name} is discontinuous at the longitude seam")
        if dataset.variables["ua_annual"].units != "m s-1" or dataset.variables["va_annual"].units != "m s-1":
            raise AssertionError(f"{path.name}: wind units are incorrect")


def main() -> None:
    manifest_path = OUT_DIR / "pmip4_forcing_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eligible_ids = {entry["forcing_id"] for entry in manifest["forcings"]}
    required_ids = {"awi_primary_3981_4000", "awi_early_3961_3980", "mpi_2330_2349"}
    if not required_ids.issubset(eligible_ids):
        raise AssertionError("A required PMIP4 forcing product is missing")
    for spec in FORCINGS:
        if spec.forcing_id not in eligible_ids:
            continue
        path = OUT_DIR / f"{spec.forcing_id}_annual_850hpa.nc"
        validate(path)
        print(f"PASS {spec.forcing_id}")
    for entry in manifest.get("ineligible_forcings", []):
        print(f"EXCLUDED {entry['forcing_id']}: {entry['reason']}")
    print("All PMIP4 forcing products passed validation.")


if __name__ == "__main__":
    main()
