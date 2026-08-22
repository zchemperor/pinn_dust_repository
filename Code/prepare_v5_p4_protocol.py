"""Create the immutable P4 forcing-sensitivity protocol manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
PROCESSED = DATA / "processed_data" / "pmip4_wind"
V5 = DATA / "model_results" / "v5_pmip4_2d"
P3_PROTOCOL = V5 / "p3_protocol" / "protocol_manifest.json"
OUTPUT = V5 / "p4_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    p3_manifest = json.loads(P3_PROTOCOL.read_text(encoding="utf-8"))
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Frozen P4 forcing sensitivity. Quantify forcing structural uncertainty; do not select the best climate model.",
        "primary_reference": "awi_primary_3981_4000_annual_850hpa.nc",
        "formal_forcings": {
            "awi_early": file_record(PROCESSED / "awi_early_3961_3980_annual_850hpa.nc"),
            "mpi": file_record(PROCESSED / "mpi_2330_2349_annual_850hpa.nc"),
        },
        "excluded_forcing": "MIROC-ES2L 850 hPa has an unresolved spatial mask; no gap filling or P4 training is permitted.",
        "frozen_model": {
            "variant": "B1_pmip4_constd",
            "network": "[2] + [32] * 5 + [1], SELU, Glorot normal",
            "learning_rate": 0.0001,
            "num_domain": 2592,
            "num_boundary": 216,
            "formal_epochs": 150000,
            "seeds": [42, 520, 2002, 2026, 3407],
        },
        "splits": p3_manifest["outer_confirmation_splits"],
        "p3_protocol_manifest": file_record(P3_PROTOCOL),
        "p4_runner": file_record(ROOT / "Code" / "v5_pmip4_2d_p3.py"),
        "p3_source_records_sha256": p3_manifest["source_sha256"],
        "formal_task_count": 30,
        "comparison_rule": "Compare each alternative forcing with the frozen AWI-primary B1 P3-B results split-by-split and seed-by-seed. Report disagreement as forcing sensitivity, not model selection.",
    }
    OUTPUT.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
