"""Frozen diffusion-only PINN ablation for the PMIP4 v5 workflow.

The runner reuses B1's network, bounded global effective diffusivity, loss
weights, boundary constraints, collocation counts, optimizer, seeds, and
train-only spatial splits.  Its sole intentional difference is that the PDE
receives U=V=0 through ``ForcingMode(..., 'zero')``.  Therefore it tests the
incremental value of the PMIP4 wind-advection terms relative to a diffusion
PDE regularizer, not a differently tuned neural architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pmip4_spherical_pde import ForcingMode, PMIP4VectorForcing
from v5_pmip4_2d_p3 import (
    FORCING_PATHS,
    OUTER_SPLITS,
    STRESS_SPLITS,
    PinnRunner,
    SplitData,
    Variant,
    load_split,
    set_seeds,
    tf,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
P2_SPLIT = DATA / "model_results" / "v5_pmip4_2d" / "p2_minimal" / "development_split"
RESULTS = DATA / "model_results" / "v5_pmip4_2d" / "b2_diffusion_only"
P3_PROTOCOL = RESULTS / "b2_diffusion_only_p3_protocol.json"
P3C_PROTOCOL = RESULTS / "b2_diffusion_only_p3c_protocol.json"
SEEDS = [42, 520, 2002, 2026, 3407]
P2_EPOCHS = 20_000
P3_EPOCHS = 150_000
NUM_DOMAIN = 2_592
NUM_BOUNDARY = 216
LEARNING_RATE = 1e-4
LOSS_WEIGHTS = (1.0, 10.0, 0.5, 0.5, 1.0, 1.0)
VARIANT = Variant("B2_diffusion_only_PINN", "constd", 1, LOSS_WEIGHTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen B2 diffusion-only PINN ablation.")
    parser.add_argument("--stage", choices=["p2", "p3"], default="p2")
    parser.add_argument("--split", choices=["p2_development", *OUTER_SPLITS, *STRESS_SPLITS], default="p2_development")
    parser.add_argument("--run-kind", choices=["dry-run", "pathcheck", "formal"], default="dry-run")
    parser.add_argument("--forcing-id", choices=["awi_primary"], default="awi_primary")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-domain", type=int, default=NUM_DOMAIN)
    parser.add_argument("--num-boundary", type=int, default=NUM_BOUNDARY)
    parser.add_argument("--log-period", type=int, default=1_000)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def p2_split() -> SplitData:
    manifest = json.loads((P2_SPLIT / "manifest.json").read_text(encoding="utf-8"))
    return SplitData(
        "p2_development",
        pd.read_csv(P2_SPLIT / "train_records.csv"),
        pd.read_csv(P2_SPLIT / "development_records.csv"),
        float(manifest["normalization"]["mean"]),
        float(manifest["normalization"]["std"]),
    )


def expected_epochs(args: argparse.Namespace) -> int:
    if args.run_kind == "dry-run":
        return 0
    if args.run_kind == "pathcheck":
        return 1
    return P2_EPOCHS if args.stage == "p2" else P3_EPOCHS


def frozen_configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "network": "[2] + [32] * 5 + [1], SELU, Glorot normal",
        "d_parameterization": "D_nd = 0.02 + 1.98 * sigmoid(raw); D = D_nd * a * U_ref",
        "forcing_mode": "zero: U=0 and V=0 inside the PDE; AWI primary U_ref remains the B1 scaling reference",
        "loss_weights": list(LOSS_WEIGHTS),
        "learning_rate": LEARNING_RATE,
        "num_domain": NUM_DOMAIN,
        "num_boundary": NUM_BOUNDARY,
        "p2_epochs": P2_EPOCHS,
        "p3_epochs": P3_EPOCHS,
        "seeds": SEEDS,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.stage == "p2" and args.split != "p2_development":
        raise ValueError("P2 is restricted to the fixed p2_development split")
    if args.stage == "p3" and args.split not in [*OUTER_SPLITS, *STRESS_SPLITS]:
        raise ValueError("P3 split is not in the frozen B2 P3-B/P3-C protocol")
    if args.lr != LEARNING_RATE or args.num_domain != NUM_DOMAIN or args.num_boundary != NUM_BOUNDARY:
        raise ValueError("B2 keeps B1's learning rate and collocation/boundary counts fixed")
    expected = expected_epochs(args)
    if args.epochs is not None and args.epochs != expected:
        raise ValueError(f"{args.stage}/{args.run_kind} requires exactly {expected} epochs")
    args.epochs = expected
    if args.stage == "p3":
        protocol = P3_PROTOCOL if args.split in OUTER_SPLITS else P3C_PROTOCOL
        if not protocol.exists():
            raise FileNotFoundError(f"{args.split} is locked until its B2 protocol is present: {protocol.name}")
        approved = json.loads(protocol.read_text(encoding="utf-8"))
        if args.split not in approved.get("allowed_splits", []):
            raise ValueError(f"{args.split} is not allowed by {protocol.name}")
        if approved.get("configuration") != frozen_configuration(args):
            raise ValueError(f"P3 configuration differs from {protocol.name}")
        forcing_path = FORCING_PATHS["awi_primary"]
        if approved.get("forcing_sha256") != sha256(forcing_path):
            raise ValueError(f"P3 forcing hash differs from {protocol.name}")


class DiffusionOnlyRunner(PinnRunner):
    """B1 runner with only the forcing mode changed to zero vector wind."""

    def reset(self) -> None:
        tf.compat.v1.reset_default_graph()
        set_seeds(self.args.seed)
        self.forcing = ForcingMode(PMIP4VectorForcing(self.forcing_path), "zero")
        self.d_raw = self.north_mean = self.south_mean = None


def protocol_payload(args: argparse.Namespace, split: SplitData) -> dict[str, object]:
    forcing_path = FORCING_PATHS["awi_primary"]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "B2 diffusion-only PINN ablation. The only intentional difference from B1 is U=V=0 inside the PDE.",
        "stage": args.stage,
        "split": split.name,
        "run_kind": args.run_kind,
        "seed": args.seed,
        "epochs": args.epochs,
        "forcing_file": str(forcing_path.relative_to(ROOT)).replace("\\", "/"),
        "forcing_sha256": sha256(forcing_path),
        "configuration": frozen_configuration(args),
        "normalization": {"method": "train_only_zscore_log_dep", "mean": split.mean, "std": split.std},
    }


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    split = p2_split() if args.stage == "p2" else load_split(args.split)
    run_id = args.run_id or f"{args.stage}_{args.split}_seed{args.seed}_{args.run_kind}"
    run_dir = RESULTS / ("p2_runs" if args.stage == "p2" else "p3_runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment_protocol.json").write_text(json.dumps(protocol_payload(args, split), indent=2), encoding="utf-8")
    if args.run_kind == "dry-run":
        payload = {
            "status": "passed", "train_rows": int(len(split.train)), "test_rows": int(len(split.test)),
            "forcing_mode": "zero", "epochs": 0, "seed": args.seed,
        }
        (run_dir / "dry_run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return run_dir
    runner = DiffusionOnlyRunner(args, VARIANT, split)
    result = runner.run(run_dir / split.name / VARIANT.name)
    result.update({"stage": args.stage, "forcing_mode": "zero", "b1_equivalence": "architecture, D bounds, loss weights, boundary constraints, collocation counts, optimizer, and seed are fixed to B1"})
    (run_dir / "b2_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return run_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
