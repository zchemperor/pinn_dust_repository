"""Materialize the immutable protocol for the V5 loss-weight robustness audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from v5_loss_weight_robustness_audit import (
    AUDIT_DIR,
    AUDIT_PROTOCOL,
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    FORCING_PATH,
    LOSS_WEIGHTS_BY_PDE_WEIGHT,
    RUNNER_SOURCE,
    SPLITS,
    split_paths,
)


ROOT = Path(__file__).resolve().parents[1]
P3_PROTOCOL = ROOT / "Data" / "model_results" / "v5_pmip4_2d" / "p3_protocol" / "protocol_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload() -> dict[str, object]:
    p3_manifest = json.loads(P3_PROTOCOL.read_text(encoding="utf-8"))
    empirical_source = ROOT / str(p3_manifest["source"])
    tasks = [
        {"pde_residual_weight": weight, "split": split, "seed": seed, "epochs": FORMAL_EPOCHS}
        for weight in LOSS_WEIGHTS_BY_PDE_WEIGHT
        for split in SPLITS
        for seed in FORMAL_SEEDS
    ]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Post hoc robustness audit of the already observed V5 P3-C diffusion-only PINN result. "
            "Only the PDE-residual loss weight is varied around the nominal value; this is not model selection "
            "or a new confirmation experiment."
        ),
        "interpretation_boundary": (
            "P3-C remains a broad-extrapolation stress diagnostic. This audit can assess whether the direction of "
            "the matched diffusion-versus-no-PDE contrast depends on the PDE loss weight, but cannot validate the "
            "source-free PDE, PMIP4 winds, global reconstruction, or calibrated predictive uncertainty."
        ),
        "comparison": {
            "reference_weight_0": "Existing matched no-PDE MLP; PDE residual weight is 0.0.",
            "reference_weight_1": "Existing frozen zero-wind diffusion-only PINN; PDE residual weight is 1.0.",
            "new_weights": [0.3, 3.0],
            "analysis_seed_rule": "Use seeds 42, 520, and 2002 for every weight, including existing references.",
        },
        "forcing": {
            "id": "awi_primary_reference_scale_only",
            "file": str(FORCING_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(FORCING_PATH),
            "mode": "zero wind inside the PDE; AWI primary reference speed remains the diffusivity scale",
        },
        "empirical_source": {
            "file": str(empirical_source.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(empirical_source),
        },
        "split_protocol": {
            "file": str(P3_PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(P3_PROTOCOL),
            "source_sha256": p3_manifest["source_sha256"],
            "splits": {
                split: {name: sha256(path) for name, path in split_paths(split).items()}
                for split in SPLITS
            },
        },
        "runner_source": {
            "file": str(RUNNER_SOURCE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(RUNNER_SOURCE),
        },
        "configuration": {
            "network": "[2] + [32] * 5 + [1], SELU, Glorot normal",
            "model": "zero-wind diffusion-only constant-D PINN",
            "d_parameterization": "D_nd = 0.02 + 1.98 * sigmoid(raw); D = D_nd * a * U_ref",
            "boundary_conditions": "periodic value and derivative at longitude edges; learned constant north and south values",
            "train_distribution": "uniform",
            "non_pde_loss_weights": [10.0, 0.5, 0.5, 1.0, 1.0],
            "loss_weights_by_pde_residual_weight": {
                str(weight): list(weights) for weight, weights in LOSS_WEIGHTS_BY_PDE_WEIGHT.items()
            },
            "learning_rate": 0.0001,
            "num_domain": 2592,
            "num_boundary": 216,
            "formal_epochs": FORMAL_EPOCHS,
            "log_period": 1000,
        },
        "formal_seeds": FORMAL_SEEDS,
        "formal_tasks": tasks,
        "prohibited_actions": [
            "changing a loss component other than the declared PDE-residual weight",
            "adding further loss weights, seeds, splits, epochs, or selective reruns after observing audit results",
            "selecting a preferred weight from P3-C or presenting this post hoc audit as prospective validation",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the frozen V5 loss-weight audit protocol.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing protocol before any formal training.")
    args = parser.parse_args()
    if AUDIT_PROTOCOL.exists() and not args.overwrite:
        raise FileExistsError(f"Protocol already exists: {AUDIT_PROTOCOL}. Refusing to overwrite it.")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PROTOCOL.write_text(json.dumps(payload(), indent=2), encoding="utf-8")
    print(AUDIT_PROTOCOL)


if __name__ == "__main__":
    main()
