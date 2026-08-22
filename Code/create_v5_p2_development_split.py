"""Create the frozen internal spatial development split for v5 P2.

This split is deliberately separate from the historical latitude-band stress
tests and from the future P3 outer confirmation blocks. It is used only for
P2 configuration selection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Data" / "processed_data" / "df_empirical_LGM.csv"
OUTPUT = ROOT / "Data" / "model_results" / "v5_pmip4_2d" / "p2_minimal" / "development_split"

LON_MIN, LON_MAX = 0.0, 60.0
LAT_MIN, LAT_MAX = 30.0, 90.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    frame = pd.read_csv(SOURCE)
    required = {"lon", "lat", "log_dep"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Source is missing required columns: {sorted(required - set(frame.columns))}")
    if len(frame) != 317:
        raise ValueError(f"Expected 317 Emp-LGM records, found {len(frame)}")

    development_mask = (
        (frame["lon"] >= LON_MIN)
        & (frame["lon"] < LON_MAX)
        & (frame["lat"] >= LAT_MIN)
        & (frame["lat"] <= LAT_MAX)
    )
    development = frame.loc[development_mask].copy()
    training = frame.loc[~development_mask].copy()
    if len(development) != 36 or len(training) != 281:
        raise ValueError(f"Unexpected split sizes: train={len(training)}, development={len(development)}")

    mean = float(training["log_dep"].mean())
    std = float(training["log_dep"].std(ddof=1))
    if not (std > 0):
        raise ValueError("Training label standard deviation must be positive")
    for subset in (training, development):
        subset["log_dep_norm"] = (subset["log_dep"] - mean) / std

    OUTPUT.mkdir(parents=True, exist_ok=True)
    training.to_csv(OUTPUT / "train_records.csv", index=False)
    development.to_csv(OUTPUT / "development_records.csv", index=False)
    manifest = {
        "purpose": "P2 configuration selection only; not a P3 outer confirmation split.",
        "split_id": "p2_dev_lon0_60_lat30_90",
        "bounds": {"lon_min_inclusive": LON_MIN, "lon_max_exclusive": LON_MAX, "lat_min_inclusive": LAT_MIN, "lat_max_inclusive": LAT_MAX},
        "n_train": int(len(training)),
        "n_development": int(len(development)),
        "normalization": {"method": "train_only_zscore_log_dep", "mean": mean, "std": std},
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256(SOURCE),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
