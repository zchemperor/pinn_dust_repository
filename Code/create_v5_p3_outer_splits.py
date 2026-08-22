"""Create frozen P3 outer and stress-test splits for the v5 PMIP4 route."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Data" / "processed_data" / "df_empirical_LGM.csv"
OUTPUT = ROOT / "Data" / "model_results" / "v5_pmip4_2d" / "p3_protocol"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def p2_development_mask(frame: pd.DataFrame) -> pd.Series:
    return (frame["lon"] >= 0.0) & (frame["lon"] < 60.0) & (frame["lat"] >= 30.0) & (frame["lat"] <= 90.0)


def split_masks(frame: pd.DataFrame) -> dict[str, tuple[str, pd.Series]]:
    return {
        "outer_north_america": ("lon < -60 and lat > 30", (frame["lon"] < -60.0) & (frame["lat"] > 30.0)),
        "outer_east_asia": ("lon >= 60 and lat > 30", (frame["lon"] >= 60.0) & (frame["lat"] > 30.0)),
        "outer_tropical_atlantic_africa": (
            "-60 <= lon < 60 and -30 < lat <= 30",
            (frame["lon"] >= -60.0) & (frame["lon"] < 60.0) & (frame["lat"] > -30.0) & (frame["lat"] <= 30.0),
        ),
        "stress_leave_south": ("lat <= -30", frame["lat"] <= -30.0),
        "stress_leave_tropics": ("-30 < lat <= 30", (frame["lat"] > -30.0) & (frame["lat"] <= 30.0)),
        "stress_leave_north_excluding_p2dev": ("lat > 30 and not P2 development block", (frame["lat"] > 30.0) & ~p2_development_mask(frame)),
    }


def write_split(name: str, rule: str, frame: pd.DataFrame, test_mask: pd.Series, development_mask: pd.Series) -> dict[str, object]:
    if bool((test_mask & development_mask).any()):
        raise ValueError(f"{name} leaks P2 development records into its test set")
    test = frame.loc[test_mask].copy()
    train = frame.loc[~test_mask].copy()
    if len(test) < 20 or len(train) < 100:
        raise ValueError(f"{name} has insufficient records: train={len(train)}, test={len(test)}")
    mean = float(train["log_dep"].mean())
    std = float(train["log_dep"].std(ddof=1))
    if not std > 0:
        raise ValueError(f"{name} has invalid training standard deviation")
    for subset in (train, test):
        subset["log_dep_norm"] = (subset["log_dep"] - mean) / std
        subset["p2_development_record"] = development_mask.loc[subset.index].to_numpy()
    directory = OUTPUT / name
    directory.mkdir(parents=True, exist_ok=True)
    train.to_csv(directory / "train_records.csv", index=False)
    test.to_csv(directory / "test_records.csv", index=False)
    manifest = {
        "split": name,
        "test_rule": rule,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "p2_development_records_in_train": int(train["p2_development_record"].sum()),
        "p2_development_records_in_test": int(test["p2_development_record"].sum()),
        "normalization": {"method": "train_only_zscore_log_dep", "mean": mean, "std": std},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    frame = pd.read_csv(SOURCE).reset_index().rename(columns={"index": "record_id"})
    if len(frame) != 317:
        raise ValueError(f"Expected 317 Emp-LGM records, found {len(frame)}")
    required = {"record_id", "lon", "lat", "log_dep"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(frame.columns))}")
    development_mask = p2_development_mask(frame)
    if int(development_mask.sum()) != 36:
        raise ValueError("P2 development block no longer contains the expected 36 records")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifests = []
    for name, (rule, test_mask) in split_masks(frame).items():
        manifests.append(write_split(name, rule, frame, test_mask, development_mask))
    protocol = {
        "purpose": "Frozen P3 outer confirmation and stress-test splits; P2 development records never enter a P3 test set.",
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256(SOURCE),
        "p2_development_rule": "0 <= lon < 60 and 30 <= lat <= 90",
        "p2_development_records": int(development_mask.sum()),
        "outer_confirmation_splits": [manifest["split"] for manifest in manifests if manifest["split"].startswith("outer_")],
        "stress_test_splits": [manifest["split"] for manifest in manifests if manifest["split"].startswith("stress_")],
        "splits": manifests,
    }
    (OUTPUT / "protocol_manifest.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
