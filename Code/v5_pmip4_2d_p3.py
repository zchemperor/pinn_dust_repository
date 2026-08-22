"""Frozen-protocol P3/P4 runner for PMIP4-2D PINN confirmation experiments.

P3-A uses this entry at 20k epochs to validate data paths and model outputs.
P3-B/P3-C reuse the same code with the predeclared 150k budget and seed list.
P4 reuses the frozen B1 configuration and outer splits for forcing sensitivity only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
PROCESSED = DATA / "processed_data"
PROTOCOL = DATA / "model_results" / "v5_pmip4_2d" / "p3_protocol"
RESULTS = DATA / "model_results" / "v5_pmip4_2d" / "p3_runs"
P4_RESULTS = DATA / "model_results" / "v5_pmip4_2d" / "p4_runs"
P4_PROTOCOL = DATA / "model_results" / "v5_pmip4_2d" / "p4_protocol.json"
B2_NOPDE_P3C_PROTOCOL = DATA / "model_results" / "v5_pmip4_2d" / "b2_nopde_p3c_protocol.json"
FORCING_PATHS = {
    "awi_primary": PROCESSED / "pmip4_wind" / "awi_primary_3981_4000_annual_850hpa.nc",
    "awi_early": PROCESSED / "pmip4_wind" / "awi_early_3961_3980_annual_850hpa.nc",
    "mpi": PROCESSED / "pmip4_wind" / "mpi_2330_2349_annual_850hpa.nc",
}

os.environ.setdefault("DDE_BACKEND", "tensorflow.compat.v1")
CUDA_ROOT = ROOT.parent / "cuda_cudnn"
if (CUDA_ROOT / "nvvm" / "libdevice" / "libdevice.10.bc").exists():
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={CUDA_ROOT}"

import deepxde as dde
from deepxde.backend import tf

from pmip4_spherical_pde import (
    DEFAULT_COORDINATES,
    EARTH_RADIUS_M,
    FLOAT_DTYPE,
    PMIP4VectorForcing,
    spherical_transport_terms,
)


dde.config.set_default_float("float64")
dde.config.disable_xla_jit()


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str
    outputs: int
    loss_weights: tuple[float, ...]


VARIANTS = {
    "b1_constd": Variant("B1_pmip4_constd", "constd", 1, (1.0, 10.0, 0.5, 0.5, 1.0, 1.0)),
    "m1_relative": Variant(
        "M1r_pmip4_relativeD",
        "relative",
        2,
        (1.0, 0.05, 0.05, 10.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0),
    ),
    "b2_nopde": Variant("B2_nopde_mlp", "nopde", 1, (0.0, 10.0, 0.5, 0.5, 1.0, 1.0)),
}
GP_NAME = "B3_gp_rbf"
OUTER_SPLITS = ["outer_north_america", "outer_east_asia", "outer_tropical_atlantic_africa"]
STRESS_SPLITS = ["stress_leave_south", "stress_leave_tropics", "stress_leave_north_excluding_p2dev"]
P3_SPLITS = [*OUTER_SPLITS, *STRESS_SPLITS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen P3 experiments or P4 forcing-sensitivity experiments.")
    parser.add_argument("--phase", choices=["p3", "p4"], default="p3")
    parser.add_argument("--forcing-id", choices=FORCING_PATHS, default="awi_primary")
    parser.add_argument("--run-kind", choices=["pathcheck", "formal"], default="pathcheck")
    parser.add_argument("--splits", choices=P3_SPLITS, nargs="+", default=OUTER_SPLITS)
    parser.add_argument("--variants", choices=[*VARIANTS, "b3_gp"], nargs="+", default=["b1_constd", "m1_relative", "b2_nopde", "b3_gp"])
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-domain", type=int, default=2592)
    parser.add_argument("--num-boundary", type=int, default=216)
    parser.add_argument("--log-period", type=int, default=1000)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def experiment_stage(splits: list[str]) -> str:
    selected = set(splits)
    if selected.issubset(OUTER_SPLITS):
        return "P3-B outer confirmation"
    if selected.issubset(STRESS_SPLITS):
        return "P3-C stress test"
    raise ValueError("Do not combine P3-B outer-confirmation and P3-C stress splits in one run")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol_args(args: argparse.Namespace) -> str:
    """Reject changes that would turn a forcing sensitivity into a new model search."""
    stage = experiment_stage(args.splits)
    if args.phase == "p3":
        if args.forcing_id != "awi_primary":
            raise ValueError("P3 is frozen to the AWI primary forcing")
        return stage
    if not P4_PROTOCOL.exists():
        raise FileNotFoundError(f"Missing frozen P4 protocol: {P4_PROTOCOL}")
    p4_protocol = json.loads(P4_PROTOCOL.read_text(encoding="utf-8"))
    if stage != "P3-B outer confirmation":
        raise ValueError("P4 is restricted to the three frozen outer-confirmation splits")
    if args.variants != ["b1_constd"]:
        raise ValueError("P4 is restricted to the frozen B1 constD winner")
    if args.forcing_id not in {"awi_early", "mpi"}:
        raise ValueError("P4 accepts only the AWI early-window or MPI forcing")
    forcing_record = p4_protocol["formal_forcings"][args.forcing_id]
    expected_path = str(FORCING_PATHS[args.forcing_id].relative_to(ROOT)).replace("\\", "/")
    if forcing_record["path"] != expected_path:
        raise ValueError("P4 forcing path does not match the frozen protocol")
    if sha256(FORCING_PATHS[args.forcing_id]) != forcing_record["sha256"]:
        raise ValueError("P4 forcing file hash does not match the frozen protocol")
    if args.lr != 1e-4 or args.num_domain != 2592 or args.num_boundary != 216:
        raise ValueError("P4 hyperparameters are frozen; only seed, split, forcing, and run ID may vary")
    expected_epochs = 1 if args.run_kind == "pathcheck" else 150_000
    if args.epochs != expected_epochs:
        raise ValueError(f"P4 {args.run_kind} runs require exactly {expected_epochs} epochs")
    return "P4 forcing sensitivity"


def validate_b2_nopde_p3c_protocol(args: argparse.Namespace) -> dict[str, object] | None:
    """Lock the additional no-PDE P3-C control without changing legacy P3 runs."""
    if not (
        args.phase == "p3"
        and set(args.splits).issubset(STRESS_SPLITS)
        and args.variants == ["b2_nopde"]
        and args.run_kind == "formal"
    ):
        return None
    if not B2_NOPDE_P3C_PROTOCOL.exists():
        raise FileNotFoundError(f"Missing frozen B2 no-PDE P3-C protocol: {B2_NOPDE_P3C_PROTOCOL}")
    protocol = json.loads(B2_NOPDE_P3C_PROTOCOL.read_text(encoding="utf-8"))
    config = protocol["configuration"]
    if args.forcing_id != protocol["forcing_id"]:
        raise ValueError("B2 no-PDE P3-C forcing differs from the frozen protocol")
    if sha256(FORCING_PATHS[args.forcing_id]) != protocol["forcing_sha256"]:
        raise ValueError("B2 no-PDE P3-C forcing hash differs from the frozen protocol")
    if not set(args.splits).issubset(set(protocol["allowed_splits"])):
        raise ValueError("B2 no-PDE P3-C split differs from the frozen protocol")
    split_manifest = json.loads((PROTOCOL / "protocol_manifest.json").read_text(encoding="utf-8"))
    if split_manifest["source_sha256"] != protocol["split_source_sha256"]:
        raise ValueError("B2 no-PDE P3-C split source hash differs from the frozen protocol")
    source_path = ROOT / split_manifest["source"]
    if sha256(source_path) != protocol["split_source_sha256"]:
        raise ValueError("B2 no-PDE P3-C empirical source hash differs from the frozen protocol")
    if args.seed not in protocol["seeds"] or args.epochs != config["epochs"]:
        raise ValueError("B2 no-PDE P3-C seed or epoch budget differs from the frozen protocol")
    if args.lr != config["learning_rate"]:
        raise ValueError("B2 no-PDE P3-C learning rate differs from the frozen protocol")
    if args.num_domain != config["num_domain"] or args.num_boundary != config["num_boundary"]:
        raise ValueError("B2 no-PDE P3-C collocation or boundary count differs from the frozen protocol")
    if list(VARIANTS["b2_nopde"].loss_weights) != config["loss_weights"]:
        raise ValueError("B2 no-PDE P3-C loss weights differ from the frozen protocol")
    return protocol


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def reference_speed(forcing_path: Path) -> float:
    with np.load(forcing_path.with_suffix(".npz")) as forcing:
        speed = np.asarray(forcing["wind_speed_annual"], dtype=np.float64)
    return float(np.sqrt(np.mean(speed**2)))


def normalized_diffusivity(raw: tf.Tensor) -> tf.Tensor:
    return 0.02 + 1.98 * tf.math.sigmoid(tf.cast(raw, FLOAT_DTYPE))


def relative_diffusivity(global_raw: tf.Tensor, local_raw: tf.Tensor) -> tf.Tensor:
    return normalized_diffusivity(global_raw) * tf.exp(0.7 * tf.tanh(tf.cast(local_raw, FLOAT_DTYPE)))


@dataclass(frozen=True)
class SplitData:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    mean: float
    std: float


def load_split(name: str) -> SplitData:
    directory = PROTOCOL / name
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    train = pd.read_csv(directory / "train_records.csv")
    test = pd.read_csv(directory / "test_records.csv")
    required = {"record_id", "lon", "lat", "log_dep", "log_dep_norm"}
    if not required.issubset(train.columns) or not required.issubset(test.columns):
        raise ValueError(f"{name}: missing required split columns")
    if int(test.get("p2_development_record", pd.Series(False)).sum()) != 0:
        raise ValueError(f"{name}: P2 development records leaked into P3 test data")
    normalization = manifest["normalization"]
    return SplitData(name, train, test, float(normalization["mean"]), float(normalization["std"]))


def raw_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted) - np.asarray(observed)
    return {
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "bias": float(np.mean(error)),
        "p95_abs_error": float(np.percentile(np.abs(error), 95)),
        "prediction_min": float(np.min(predicted)),
        "prediction_max": float(np.max(predicted)),
        "prediction_finite_fraction": float(np.isfinite(predicted).mean()),
    }


class PinnRunner:
    def __init__(self, args: argparse.Namespace, variant: Variant, split: SplitData):
        self.args, self.variant, self.split = args, variant, split
        self.geometry = dde.geometry.geometry_2d.Rectangle(np.array([-2.0, -0.89]), np.array([2.0, 0.89]))
        self.forcing_path = FORCING_PATHS[args.forcing_id]
        self.reference_speed = reference_speed(self.forcing_path)
        self.diffusivity_scale = EARTH_RADIUS_M * self.reference_speed
        self.forcing: PMIP4VectorForcing | None = None
        self.d_raw = None
        self.north_mean = None
        self.south_mean = None

    def reset(self) -> None:
        tf.compat.v1.reset_default_graph()
        set_seeds(self.args.seed)
        self.forcing = PMIP4VectorForcing(self.forcing_path)
        self.d_raw = self.north_mean = self.south_mean = None

    @staticmethod
    def north(x, on_boundary):
        return on_boundary and np.isclose(x[1], 0.89)

    @staticmethod
    def south(x, on_boundary):
        return on_boundary and np.isclose(x[1], -0.89)

    @staticmethod
    def periodic(x, on_boundary):
        return on_boundary and (np.isclose(x[0], -2.0) or np.isclose(x[0], 2.0))

    def smoothness(self, value: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        gradient = tf.gradients(value, x, unconnected_gradients="zero")[0]
        d_lon = gradient[:, 0:1] / DEFAULT_COORDINATES.lon_radians_per_unit
        d_lat = gradient[:, 1:2] / DEFAULT_COORDINATES.lat_radians_per_unit
        return d_lon**2 / tf.cos(DEFAULT_COORDINATES.latitude(x)) ** 2 + d_lat**2

    def pde_constd(self, x, y):
        if self.forcing is None or self.d_raw is None:
            raise RuntimeError("constant-D state is not initialized")
        d = normalized_diffusivity(self.d_raw) * self.diffusivity_scale
        return spherical_transport_terms(x, y[:, 0:1], d, self.forcing)["total"] * EARTH_RADIUS_M / self.reference_speed

    def pde_relative(self, x, y):
        if self.forcing is None or self.d_raw is None:
            raise RuntimeError("relative-D state is not initialized")
        local_raw = y[:, 1:2]
        d = relative_diffusivity(self.d_raw, local_raw) * self.diffusivity_scale
        residual = spherical_transport_terms(x, y[:, 0:1], d, self.forcing)["total"] * EARTH_RADIUS_M / self.reference_speed
        zero_mean = tf.ones_like(local_raw) * tf.reduce_mean(local_raw)
        return [residual, self.smoothness(local_raw, x), zero_mean]

    @staticmethod
    def pde_zero(x, y):
        return tf.zeros_like(y[:, 0:1])

    def build(self):
        observation = dde.icbc.PointSetBC(
            self.split.train[["lon", "lat"]].to_numpy(dtype=np.float64) / 90.0,
            self.split.train["log_dep_norm"].to_numpy(dtype=np.float64).reshape(-1, 1),
            component=0,
        )
        self.north_mean, self.south_mean = dde.Variable(-1.0), dde.Variable(-2.0)
        u_bcs = [
            dde.icbc.PeriodicBC(self.geometry, component_x=0, on_boundary=self.periodic, derivative_order=0, component=0),
            dde.icbc.PeriodicBC(self.geometry, component_x=0, on_boundary=self.periodic, derivative_order=1, component=0),
            dde.DirichletBC(self.geometry, lambda _: self.north_mean, self.north),
            dde.DirichletBC(self.geometry, lambda _: self.south_mean, self.south),
        ]
        external = [self.north_mean, self.south_mean]
        if self.variant.kind == "relative":
            self.d_raw = dde.Variable(-1.0)
            external.insert(0, self.d_raw)
            d_bcs = [
                dde.icbc.PeriodicBC(self.geometry, component_x=0, on_boundary=self.periodic, derivative_order=0, component=1),
                dde.icbc.PeriodicBC(self.geometry, component_x=0, on_boundary=self.periodic, derivative_order=1, component=1),
            ]
            pde, bcs = self.pde_relative, [observation, u_bcs[0], u_bcs[1], d_bcs[0], d_bcs[1], u_bcs[2], u_bcs[3]]
        elif self.variant.kind == "constd":
            self.d_raw = dde.Variable(-1.0)
            external.insert(0, self.d_raw)
            pde, bcs = self.pde_constd, [observation, *u_bcs]
        else:
            pde, bcs = self.pde_zero, [observation, *u_bcs]
        data = dde.data.PDE(
            self.geometry, pde, bcs, num_domain=self.args.num_domain, num_boundary=self.args.num_boundary,
            anchors=self.split.train[["lon", "lat"]].to_numpy(dtype=np.float64) / 90.0, train_distribution="uniform",
        )
        net = dde.maps.FNN([2] + [32] * 5 + [self.variant.outputs], "selu", "Glorot normal")
        model = dde.Model(data, net)
        model.compile("adam", lr=self.args.lr, loss_weights=self.variant.loss_weights, external_trainable_variables=external)
        return model

    @staticmethod
    def save_history(history, directory: Path) -> None:
        train, test = np.asarray(history.loss_train), np.asarray(history.loss_test)
        values: dict[str, object] = {"step": history.steps}
        for index in range(train.shape[1]):
            values[f"train_loss_{index}"] = train[:, index]
        for index in range(test.shape[1]):
            values[f"test_loss_{index}"] = test[:, index]
        pd.DataFrame(values).to_csv(directory / "loss_history.csv", index=False)

    def run(self, directory: Path) -> dict[str, object]:
        self.reset()
        directory.mkdir(parents=True, exist_ok=True)
        model = self.build()
        variables = [self.north_mean, self.south_mean] if self.d_raw is None else [self.d_raw, self.north_mean, self.south_mean]
        callbacks = [
            dde.callbacks.ModelCheckpoint(str(directory / "model.ckpt"), verbose=0, period=self.args.log_period, save_better_only=False),
            dde.callbacks.VariableValue(variables, period=self.args.log_period, filename=str(directory / "variables.dat")),
        ]
        started = time.perf_counter()
        history, state = model.train(epochs=self.args.epochs, display_every=self.args.log_period, callbacks=callbacks)
        elapsed = time.perf_counter() - started
        model.saver.save(model.sess, str(directory / "model.ckpt"), global_step=model.train_state.step)
        self.save_history(history, directory)
        test_x = self.split.test[["lon", "lat"]].to_numpy(dtype=np.float64) / 90.0
        prediction = model.predict(test_x)
        predicted_raw = prediction[:, 0] * self.split.std + self.split.mean
        test_out = self.split.test.copy()
        test_out["prediction_log_dep_norm"] = prediction[:, 0]
        test_out["prediction_log_dep"] = predicted_raw
        test_out["error_log_dep"] = predicted_raw - test_out["log_dep"]
        if self.variant.kind == "constd":
            with model.sess.as_default():
                d_value = float(normalized_diffusivity(self.d_raw).eval())
            test_out["d_nd"] = d_value
        elif self.variant.kind == "relative":
            with model.sess.as_default():
                baseline = float(normalized_diffusivity(self.d_raw).eval())
            test_out["d_nd"] = baseline * np.exp(0.7 * np.tanh(prediction[:, 1]))
        test_out.to_csv(directory / "test_predictions.csv", index=False)
        grid = pd.read_csv(PROCESSED / "df_global_grid.csv")
        grid_prediction = model.predict(grid[["lon", "lat"]].to_numpy(dtype=np.float64) / 90.0)
        grid_out = grid.copy()
        grid_out["prediction_log_dep_norm"] = grid_prediction[:, 0]
        grid_out["prediction_log_dep"] = grid_prediction[:, 0] * self.split.std + self.split.mean
        if self.variant.kind == "constd":
            grid_out["d_nd"] = test_out["d_nd"].iloc[0]
        elif self.variant.kind == "relative":
            grid_out["d_nd"] = baseline * np.exp(0.7 * np.tanh(grid_prediction[:, 1]))
        grid_out.to_csv(directory / "global_grid_predictions.csv", index=False)
        result: dict[str, object] = {
            "split": self.split.name, "model": self.variant.name, "seed": self.args.seed,
            "epochs_requested": self.args.epochs, "epochs_completed": int(model.train_state.step),
            "best_step": int(state.best_step), "elapsed_seconds": elapsed,
            "last_train_loss": [float(value) for value in history.loss_train[-1]],
            **raw_metrics(test_out["log_dep"].to_numpy(), predicted_raw),
        }
        if self.variant.kind == "constd":
            result["d_nd_constant"] = d_value
        elif self.variant.kind == "relative":
            result.update({"d_nd_baseline": baseline, "d_nd_min": float(test_out.d_nd.min()), "d_nd_max": float(test_out.d_nd.max())})
        (directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        model.sess.close()
        return result


def run_gp(split: SplitData, seed: int, directory: Path) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed, n_restarts_optimizer=2)
    started = time.perf_counter()
    model.fit(split.train[["lon", "lat"]].to_numpy() / 90.0, split.train["log_dep"].to_numpy())
    predicted, std = model.predict(split.test[["lon", "lat"]].to_numpy() / 90.0, return_std=True)
    elapsed = time.perf_counter() - started
    out = split.test.copy()
    out["prediction_log_dep"] = predicted
    out["prediction_std_log_dep"] = std
    out["error_log_dep"] = predicted - out["log_dep"]
    out.to_csv(directory / "test_predictions.csv", index=False)
    grid = pd.read_csv(PROCESSED / "df_global_grid.csv")
    grid_prediction, grid_std = model.predict(grid[["lon", "lat"]].to_numpy() / 90.0, return_std=True)
    grid["prediction_log_dep"], grid["prediction_std_log_dep"] = grid_prediction, grid_std
    grid.to_csv(directory / "global_grid_predictions.csv", index=False)
    result: dict[str, object] = {"split": split.name, "model": GP_NAME, "seed": seed, "elapsed_seconds": elapsed, **raw_metrics(out["log_dep"], predicted)}
    (directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    stage = validate_protocol_args(args)
    b2_nopde_protocol = validate_b2_nopde_p3c_protocol(args)
    run_id = args.run_id or f"{args.phase}_{args.forcing_id}_seed{args.seed}_{args.epochs}epochs"
    root = (P4_RESULTS if args.phase == "p4" else RESULTS) / run_id
    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "purpose": f"Frozen {stage} run.",
        "splits": args.splits, "variants": args.variants, "seed": args.seed, "epochs": args.epochs,
        "lr": args.lr, "num_domain": args.num_domain, "num_boundary": args.num_boundary,
        "phase": args.phase, "run_kind": args.run_kind,
        "network": "[2] + [32] * 5 + [outputs], SELU, Glorot normal",
        "forcing_id": args.forcing_id, "forcing": str(FORCING_PATHS[args.forcing_id].relative_to(ROOT)).replace("\\", "/"),
        "p3_protocol_manifest": str((PROTOCOL / "protocol_manifest.json").relative_to(ROOT)).replace("\\", "/"),
    }
    if args.phase == "p4":
        protocol["p4_protocol_manifest"] = str(P4_PROTOCOL.relative_to(ROOT)).replace("\\", "/")
    if b2_nopde_protocol is not None:
        protocol["frozen_control_protocol"] = str(B2_NOPDE_P3C_PROTOCOL.relative_to(ROOT)).replace("\\", "/")
        protocol["frozen_control_protocol_sha256"] = sha256(B2_NOPDE_P3C_PROTOCOL)
    (root / "experiment_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows = []
    for split_name in args.splits:
        split = load_split(split_name)
        for key in args.variants:
            if key == "b3_gp":
                print(f"[{split_name}] training {GP_NAME}", flush=True)
                rows.append(run_gp(split, args.seed, root / split_name / GP_NAME))
            else:
                print(f"[{split_name}] training {VARIANTS[key].name}", flush=True)
                rows.append(PinnRunner(args, VARIANTS[key], split).run(root / split_name / VARIANTS[key].name))
    pd.DataFrame(rows).to_csv(root / f"{args.phase}_metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
