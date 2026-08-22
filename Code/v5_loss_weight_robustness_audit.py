"""Isolated runner for the minimal V5 PDE-loss-weight robustness audit.

This runner reuses the frozen zero-wind diffusion-only PINN and varies only
the first DeepXDE loss weight (the PDE residual). Formal tasks are accepted
only when they exactly match the immutable audit protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from v5_pmip4_2d_diffusion_only import DiffusionOnlyRunner
from v5_pmip4_2d_p3 import FORCING_PATHS, STRESS_SPLITS, Variant, load_split


ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "Data" / "model_results" / "v5_pmip4_2d"
AUDIT_DIR = V5 / "loss_weight_robustness_audit"
AUDIT_PROTOCOL = AUDIT_DIR / "audit_protocol.json"
RUNNER_SOURCE = Path(__file__).resolve()
FORCING_PATH = FORCING_PATHS["awi_primary"]
SPLITS = ["stress_leave_south", "stress_leave_tropics", "stress_leave_north_excluding_p2dev"]
FORMAL_SEEDS = [42, 520, 2002]
FORMAL_EPOCHS = 150_000
LEARNING_RATE = 1e-4
NUM_DOMAIN = 2_592
NUM_BOUNDARY = 216
LOG_PERIOD = 1_000
LOSS_WEIGHTS_BY_PDE_WEIGHT = {
    0.3: (0.3, 10.0, 0.5, 0.5, 1.0, 1.0),
    3.0: (3.0, 10.0, 0.5, 0.5, 1.0, 1.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_paths(split: str) -> dict[str, Path]:
    directory = V5 / "p3_protocol" / split
    return {
        "manifest_sha256": directory / "manifest.json",
        "train_records_sha256": directory / "train_records.csv",
        "test_records_sha256": directory / "test_records.csv",
    }


def weight_tag(weight: float) -> str:
    return f"w{weight:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen V5 loss-weight robustness audit.")
    parser.add_argument("--run-kind", choices=["dry-run", "pathcheck", "formal"], default="dry-run")
    parser.add_argument("--forcing-id", choices=["awi_primary"], default="awi_primary")
    parser.add_argument("--split", choices=SPLITS, default=SPLITS[0])
    parser.add_argument("--pde-weight", type=float, choices=list(LOSS_WEIGHTS_BY_PDE_WEIGHT), default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-domain", type=int, default=NUM_DOMAIN)
    parser.add_argument("--num-boundary", type=int, default=NUM_BOUNDARY)
    parser.add_argument("--log-period", type=int, default=LOG_PERIOD)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def formal_configuration() -> dict[str, object]:
    return {
        "network": "[2] + [32] * 5 + [1], SELU, Glorot normal",
        "model": "zero-wind diffusion-only constant-D PINN",
        "d_parameterization": "D_nd = 0.02 + 1.98 * sigmoid(raw); D = D_nd * a * U_ref",
        "boundary_conditions": "periodic value and derivative at longitude edges; learned constant north and south values",
        "train_distribution": "uniform",
        "non_pde_loss_weights": [10.0, 0.5, 0.5, 1.0, 1.0],
        "loss_weights_by_pde_residual_weight": {
            str(weight): list(weights) for weight, weights in LOSS_WEIGHTS_BY_PDE_WEIGHT.items()
        },
        "learning_rate": LEARNING_RATE,
        "num_domain": NUM_DOMAIN,
        "num_boundary": NUM_BOUNDARY,
        "formal_epochs": FORMAL_EPOCHS,
        "log_period": LOG_PERIOD,
    }


def expected_epochs(run_kind: str) -> int:
    return {"dry-run": 0, "pathcheck": 1, "formal": FORMAL_EPOCHS}[run_kind]


def read_protocol() -> dict[str, object]:
    if not AUDIT_PROTOCOL.exists():
        raise FileNotFoundError(f"Missing frozen audit protocol: {AUDIT_PROTOCOL}")
    return json.loads(AUDIT_PROTOCOL.read_text(encoding="utf-8"))


def validate_protocol(args: argparse.Namespace) -> dict[str, object]:
    if args.split not in STRESS_SPLITS:
        raise ValueError("The audit is restricted to the three frozen P3-C stress splits")
    if args.forcing_id != "awi_primary":
        raise ValueError("The audit is frozen to the AWI primary reference-scale forcing")
    if args.lr != LEARNING_RATE or args.num_domain != NUM_DOMAIN or args.num_boundary != NUM_BOUNDARY:
        raise ValueError("Learning rate and collocation/boundary counts are frozen")
    if args.log_period != LOG_PERIOD:
        raise ValueError("Log period is frozen")
    expected = expected_epochs(args.run_kind)
    if args.epochs is not None and args.epochs != expected:
        raise ValueError(f"{args.run_kind} requires exactly {expected} epochs")
    args.epochs = expected
    protocol = read_protocol()
    if protocol.get("configuration") != formal_configuration():
        raise ValueError("Audit configuration differs from the frozen protocol")
    if protocol.get("forcing", {}).get("sha256") != sha256(FORCING_PATH):
        raise ValueError("AWI primary forcing hash differs from the frozen protocol")
    if protocol.get("runner_source", {}).get("sha256") != sha256(RUNNER_SOURCE):
        raise ValueError("Runner source hash differs from the frozen protocol")
    p3_manifest_path = V5 / "p3_protocol" / "protocol_manifest.json"
    p3_manifest = json.loads(p3_manifest_path.read_text(encoding="utf-8"))
    source_path = ROOT / str(p3_manifest["source"])
    if protocol.get("empirical_source", {}).get("sha256") != sha256(source_path):
        raise ValueError("Empirical source hash differs from the frozen protocol")
    split_protocol = protocol.get("split_protocol", {})
    if split_protocol.get("sha256") != sha256(p3_manifest_path):
        raise ValueError("P3 split protocol hash differs from the frozen audit protocol")
    expected_split_hashes = split_protocol.get("splits", {}).get(args.split)
    actual_split_hashes = {name: sha256(path) for name, path in split_paths(args.split).items()}
    if expected_split_hashes != actual_split_hashes:
        raise ValueError("One or more frozen P3-C split files differ from the audit protocol")
    if args.run_kind == "formal":
        task = {"pde_residual_weight": args.pde_weight, "split": args.split, "seed": args.seed, "epochs": args.epochs}
        if task not in protocol.get("formal_tasks", []):
            raise ValueError("Formal task is not in the frozen audit matrix")
    elif args.seed not in FORMAL_SEEDS:
        raise ValueError("Dry-run and pathcheck use only the three declared audit seeds")
    return protocol


def run(args: argparse.Namespace) -> Path:
    protocol = validate_protocol(args)
    run_id = args.run_id or f"{args.run_kind}_{weight_tag(args.pde_weight)}_{args.split}_seed{args.seed}"
    run_dir = AUDIT_DIR / ("formal_runs" if args.run_kind == "formal" else "pathchecks") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_protocol": str(AUDIT_PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "audit_protocol_sha256": sha256(AUDIT_PROTOCOL),
        "run_kind": args.run_kind,
        "split": args.split,
        "seed": args.seed,
        "epochs": args.epochs,
        "pde_residual_weight": args.pde_weight,
        "loss_weights": list(LOSS_WEIGHTS_BY_PDE_WEIGHT[args.pde_weight]),
        "forcing_mode": "zero wind inside PDE",
        "protocol_purpose": protocol["purpose"],
    }
    (run_dir / "experiment_protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.run_kind == "dry-run":
        split = load_split(args.split)
        summary = {"status": "passed", "train_rows": len(split.train), "test_rows": len(split.test), **metadata}
        (run_dir / "dry_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return run_dir
    split = load_split(args.split)
    variant = Variant(
        f"V5_loss_weight_audit_{weight_tag(args.pde_weight)}",
        "constd",
        1,
        LOSS_WEIGHTS_BY_PDE_WEIGHT[args.pde_weight],
    )
    runner = DiffusionOnlyRunner(args, variant, split)
    result = runner.run(run_dir / args.split / variant.name)
    result.update(metadata)
    result["pde_residual_weight"] = args.pde_weight
    (run_dir / "audit_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return run_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
