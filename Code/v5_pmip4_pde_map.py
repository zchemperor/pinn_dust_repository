"""Static low-freedom PDE-MAP baseline for the frozen PMIP4 v5 workflow.

This is deliberately independent of the legacy ERA5 ``pde_map_lite`` code.
It uses the AWI-primary PMIP4 forcing, P2/P3 train-only normalization, the
same active spherical domain as the constD PINN, periodic longitude, and
constant north/south boundaries.  A 12 x 24 periodic bilinear control lattice
represents the normalized deposition field.  Its two polar rows are each
collapsed to one constant, so the field has 242 parameters rather than a
neural-network parameterization.  The scalar diffusivity is fitted inside
the same bounded D/(a U_ref) interval used by B1.

The code has two stages.  P2 is for numerical acceptance only.  P3 is refused
until a P2-approved protocol JSON has been created manually, preventing P3
outcomes from being used to alter the MAP configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix, diags, eye, kron, vstack
from scipy.sparse.linalg import lsmr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
PROCESSED = DATA / "processed_data"
P3_PROTOCOL = DATA / "model_results" / "v5_pmip4_2d" / "p3_protocol"
P2_SPLIT = DATA / "model_results" / "v5_pmip4_2d" / "p2_minimal" / "development_split"
RESULTS = DATA / "model_results" / "v5_pmip4_2d" / "pde_map"
FORCING_PATH = PROCESSED / "pmip4_wind" / "awi_primary_3981_4000_annual_850hpa.npz"
APPROVED_PROTOCOL = RESULTS / "pde_map_p3_protocol.json"
STRESS_PROTOCOL = RESULTS / "pde_map_p3c_protocol.json"

OUTER_SPLITS = ["outer_north_america", "outer_east_asia", "outer_tropical_atlantic_africa"]
STRESS_SPLITS = ["stress_leave_south", "stress_leave_tropics", "stress_leave_north_excluding_p2dev"]
ACTIVE_LATITUDE_DEGREES = 80.1
CONTROL_LATITUDE_COUNT = 12
CONTROL_LONGITUDE_COUNT = 24
D_ND_MIN = 0.02
D_ND_MAX = 2.0
OBSERVATION_WEIGHT = 10.0
PDE_WEIGHT = 1.0
NUMERICAL_RIDGE = 1.0e-8
OUTER_XATOL = 1.0e-3
OUTER_MAXITER = 40
LSMR_MAXITER = 8_000


@dataclass(frozen=True)
class SplitData:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    mean: float
    std: float
    stage: str


@dataclass(frozen=True)
class MapSystem:
    grid_lon_deg: np.ndarray
    grid_lat_deg: np.ndarray
    grid_basis: csr_matrix
    advection_operator: csr_matrix
    diffusion_operator: csr_matrix
    n_pde_rows: int
    reference_speed: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PMIP4 low-freedom static PDE-MAP baseline.")
    parser.add_argument("--stage", choices=["p2", "p3", "p3c"], default="p2")
    parser.add_argument("--split", choices=["p2_development", *OUTER_SPLITS, *STRESS_SPLITS], default="p2_development")
    parser.add_argument("--run-kind", choices=["dry-run", "formal"], default="dry-run")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--control-lat", type=int, default=CONTROL_LATITUDE_COUNT)
    parser.add_argument("--control-lon", type=int, default=CONTROL_LONGITUDE_COUNT)
    parser.add_argument("--d-min", type=float, default=D_ND_MIN)
    parser.add_argument("--d-max", type=float, default=D_ND_MAX)
    parser.add_argument("--observation-weight", type=float, default=OBSERVATION_WEIGHT)
    parser.add_argument("--pde-weight", type=float, default=PDE_WEIGHT)
    parser.add_argument("--numerical-ridge", type=float, default=NUMERICAL_RIDGE)
    parser.add_argument("--outer-xatol", type=float, default=OUTER_XATOL)
    parser.add_argument("--outer-maxiter", type=int, default=OUTER_MAXITER)
    parser.add_argument("--lsmr-maxiter", type=int, default=LSMR_MAXITER)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_p2_split() -> SplitData:
    manifest = json.loads((P2_SPLIT / "manifest.json").read_text(encoding="utf-8"))
    train = pd.read_csv(P2_SPLIT / "train_records.csv")
    test = pd.read_csv(P2_SPLIT / "development_records.csv")
    return SplitData(
        name="p2_development",
        train=train,
        test=test,
        mean=float(manifest["normalization"]["mean"]),
        std=float(manifest["normalization"]["std"]),
        stage="P2 development",
    )


def load_p3_split(name: str, stage: str) -> SplitData:
    directory = P3_PROTOCOL / name
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    train = pd.read_csv(directory / "train_records.csv")
    test = pd.read_csv(directory / "test_records.csv")
    if int(test.get("p2_development_record", pd.Series(False)).sum()) != 0:
        raise ValueError(f"{name}: P2 development records leaked into a P3 test partition")
    return SplitData(
        name=name,
        train=train,
        test=test,
        mean=float(manifest["normalization"]["mean"]),
        std=float(manifest["normalization"]["std"]),
        stage="P3-B outer confirmation" if stage == "p3" else "P3-C stress test",
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.stage == "p2" and args.split != "p2_development":
        raise ValueError("P2 accepts only the frozen p2_development split")
    if args.stage == "p3" and args.split not in OUTER_SPLITS:
        raise ValueError("P3 accepts only frozen outer-confirmation splits")
    if args.stage == "p3c" and args.split not in STRESS_SPLITS:
        raise ValueError("P3-C accepts only frozen stress-test splits")
    if not (args.control_lat >= 4 and args.control_lon >= 4):
        raise ValueError("Control lattice must have at least 4 latitude and longitude knots")
    if not (0 < args.d_min < args.d_max):
        raise ValueError("Invalid bounded diffusivity interval")
    if args.observation_weight <= 0 or args.pde_weight <= 0 or args.numerical_ridge < 0:
        raise ValueError("Objective weights must be positive; numerical ridge may be zero")
    if args.outer_maxiter < 1 or args.lsmr_maxiter < 1 or args.outer_xatol <= 0:
        raise ValueError("Invalid optimizer limits")
    if args.stage in {"p3", "p3c"}:
        protocol_path = APPROVED_PROTOCOL if args.stage == "p3" else STRESS_PROTOCOL
        if not protocol_path.exists():
            raise FileNotFoundError(
                f"{args.stage} is locked until its approved protocol is created. "
                "Do not use held-out results to select MAP settings."
            )
        approved = json.loads(protocol_path.read_text(encoding="utf-8"))
        expected = frozen_configuration(args)
        if approved.get("configuration") != expected:
            raise ValueError(f"{args.stage} arguments do not match the approved PDE-MAP protocol")
        if approved.get("forcing_sha256") != sha256(FORCING_PATH):
            raise ValueError(f"{args.stage} forcing hash does not match the approved PDE-MAP protocol")


def frozen_configuration(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "control_lat": args.control_lat,
        "control_lon": args.control_lon,
        "d_min": args.d_min,
        "d_max": args.d_max,
        "observation_weight": args.observation_weight,
        "pde_weight": args.pde_weight,
        "numerical_ridge": args.numerical_ridge,
        "outer_xatol": args.outer_xatol,
        "outer_maxiter": args.outer_maxiter,
        "lsmr_maxiter": args.lsmr_maxiter,
    }


def control_parameter_count(n_lat: int, n_lon: int) -> int:
    return 2 + (n_lat - 2) * n_lon


def control_index(lat_index: int, lon_index: int, n_lat: int, n_lon: int) -> int:
    if lat_index == 0:
        return 0
    if lat_index == n_lat - 1:
        return 1
    return 2 + (lat_index - 1) * n_lon + (lon_index % n_lon)


def bilinear_control_basis(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    n_control_lat: int,
    n_control_lon: int,
) -> csr_matrix:
    """Return periodic bilinear interpolation from control parameters to points."""
    lon = np.asarray(lon_deg, dtype=np.float64).ravel()
    lat = np.asarray(lat_deg, dtype=np.float64).ravel()
    if lon.shape != lat.shape:
        raise ValueError("Longitude and latitude arrays must have the same shape")
    if np.any(lat < -ACTIVE_LATITUDE_DEGREES - 1e-10) or np.any(lat > ACTIVE_LATITUDE_DEGREES + 1e-10):
        raise ValueError("Evaluation points lie outside the active latitude domain")
    n_points = lon.size
    n_parameters = control_parameter_count(n_control_lat, n_control_lon)
    lat_coord = (lat + ACTIVE_LATITUDE_DEGREES) / (2.0 * ACTIVE_LATITUDE_DEGREES) * (n_control_lat - 1)
    lat_coord = np.clip(lat_coord, 0.0, n_control_lat - 1)
    lat_low = np.floor(lat_coord).astype(int)
    lat_low = np.minimum(lat_low, n_control_lat - 2)
    lat_fraction = lat_coord - lat_low
    lat_fraction[np.isclose(lat, ACTIVE_LATITUDE_DEGREES)] = 1.0

    lon_wrapped = np.mod(lon + 180.0, 360.0)
    lon_coord = lon_wrapped / 360.0 * n_control_lon
    lon_low = np.floor(lon_coord).astype(int) % n_control_lon
    lon_fraction = lon_coord - np.floor(lon_coord)
    lon_high = (lon_low + 1) % n_control_lon
    lat_high = lat_low + 1

    rows, columns, values = [], [], []
    for row in range(n_points):
        weights = (
            (lat_low[row], lon_low[row], (1.0 - lat_fraction[row]) * (1.0 - lon_fraction[row])),
            (lat_low[row], lon_high[row], (1.0 - lat_fraction[row]) * lon_fraction[row]),
            (lat_high[row], lon_low[row], lat_fraction[row] * (1.0 - lon_fraction[row])),
            (lat_high[row], lon_high[row], lat_fraction[row] * lon_fraction[row]),
        )
        for lat_index, lon_index, weight in weights:
            rows.append(row)
            columns.append(control_index(lat_index, lon_index, n_control_lat, n_control_lon))
            values.append(weight)
    return csr_matrix((values, (rows, columns)), shape=(n_points, n_parameters))


def periodic_first_derivative(n: int, spacing: float) -> csr_matrix:
    rows = np.repeat(np.arange(n), 2)
    columns = np.empty(2 * n, dtype=int)
    columns[0::2] = (np.arange(n) - 1) % n
    columns[1::2] = (np.arange(n) + 1) % n
    values = np.tile(np.array([-0.5 / spacing, 0.5 / spacing]), n)
    return csr_matrix((values, (rows, columns)), shape=(n, n))


def latitude_first_derivative(n: int, spacing: float) -> csr_matrix:
    """Second-order one-sided boundaries and centered interior differences."""
    rows, columns, values = [], [], []
    for index in range(n):
        if index == 0:
            entries = ((0, -1.5), (1, 2.0), (2, -0.5))
        elif index == n - 1:
            entries = ((n - 3, 0.5), (n - 2, -2.0), (n - 1, 1.5))
        else:
            entries = ((index - 1, -0.5), (index + 1, 0.5))
        for column, coefficient in entries:
            rows.append(index)
            columns.append(column)
            values.append(coefficient / spacing)
    return csr_matrix((values, (rows, columns)), shape=(n, n))


def build_transport_operators(lon_deg: np.ndarray, lat_deg: np.ndarray, u: np.ndarray, v: np.ndarray, reference_speed: float) -> tuple[csr_matrix, csr_matrix]:
    """Dimensionless finite-difference advection and constD diffusion matrices."""
    n_lat, n_lon = u.shape
    if v.shape != u.shape or n_lat != lat_deg.size or n_lon != lon_deg.size:
        raise ValueError("Forcing and coordinate dimensions do not agree")
    lon_rad = np.deg2rad(lon_deg)
    lat_rad = np.deg2rad(lat_deg)
    d_lon = float(np.mean(np.diff(lon_rad)))
    d_lat = float(np.mean(np.diff(lat_rad)))
    if not (np.allclose(np.diff(lon_rad), d_lon) and np.allclose(np.diff(lat_rad), d_lat)):
        raise ValueError("The static MAP requires regular forcing coordinates")
    d_lon_matrix = periodic_first_derivative(n_lon, d_lon)
    d_lat_matrix = latitude_first_derivative(n_lat, d_lat)
    d_lambda = kron(eye(n_lat, format="csr"), d_lon_matrix, format="csr")
    d_phi = kron(d_lat_matrix, eye(n_lon, format="csr"), format="csr")
    cos_phi = np.repeat(np.cos(lat_rad), n_lon)
    if np.any(np.abs(cos_phi) < 1e-8):
        raise ValueError("Do not evaluate the transport operator at the geographic poles")
    u_nd = np.asarray(u, dtype=np.float64).ravel() / reference_speed
    v_nd = np.asarray(v, dtype=np.float64).ravel() / reference_speed
    inv_cos = diags(1.0 / cos_phi, format="csr")
    cos = diags(cos_phi, format="csr")
    advection = -inv_cos @ (d_lambda @ diags(u_nd, format="csr") + d_phi @ diags(v_nd * cos_phi, format="csr"))
    diffusion = inv_cos @ (d_lambda @ inv_cos @ d_lambda + d_phi @ cos @ d_phi)
    interior_rows = np.arange(1, n_lat - 1)
    interior_indices = (interior_rows[:, None] * n_lon + np.arange(n_lon)[None, :]).ravel()
    return advection[interior_indices, :].tocsr(), diffusion[interior_indices, :].tocsr()


def load_map_system(args: argparse.Namespace) -> MapSystem:
    with np.load(FORCING_PATH) as forcing:
        source_lon = np.asarray(forcing["lon"], dtype=np.float64)
        source_lat = np.asarray(forcing["lat"], dtype=np.float64)
        source_u = np.asarray(forcing["ua_annual"], dtype=np.float64)
        source_v = np.asarray(forcing["va_annual"], dtype=np.float64)
        speed = np.asarray(forcing["wind_speed_annual"], dtype=np.float64)
    active = (source_lat >= -ACTIVE_LATITUDE_DEGREES) & (source_lat <= ACTIVE_LATITUDE_DEGREES)
    grid_lat = source_lat[active]
    grid_lon = source_lon
    u, v = source_u[active, :], source_v[active, :]
    reference_speed = float(np.sqrt(np.mean(speed**2)))
    if not (np.isfinite(u).all() and np.isfinite(v).all() and np.isfinite(reference_speed) and reference_speed > 0):
        raise ValueError("Forcing contains invalid values")
    lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat, indexing="xy")
    basis = bilinear_control_basis(lon_grid.ravel(), lat_grid.ravel(), args.control_lat, args.control_lon)
    advection, diffusion = build_transport_operators(grid_lon, grid_lat, u, v, reference_speed)
    return MapSystem(
        grid_lon_deg=grid_lon,
        grid_lat_deg=grid_lat,
        grid_basis=basis,
        advection_operator=advection,
        diffusion_operator=diffusion,
        n_pde_rows=advection.shape[0],
        reference_speed=reference_speed,
    )


class StaticPdeMap:
    def __init__(self, args: argparse.Namespace, split: SplitData, system: MapSystem):
        self.args, self.split, self.system = args, split, system
        self.observation_basis = bilinear_control_basis(
            split.train["lon"].to_numpy(), split.train["lat"].to_numpy(), args.control_lat, args.control_lon
        )
        self.trace: list[dict[str, float | int]] = []

    def objective_matrix(self, d_nd: float) -> tuple[csr_matrix, np.ndarray]:
        pde = (self.system.advection_operator + d_nd * self.system.diffusion_operator) @ self.system.grid_basis
        observation_scale = np.sqrt(self.args.observation_weight / self.observation_basis.shape[0])
        pde_scale = np.sqrt(self.args.pde_weight / self.system.n_pde_rows)
        blocks = [observation_scale * self.observation_basis, pde_scale * pde]
        target = [observation_scale * self.split.train["log_dep_norm"].to_numpy(dtype=np.float64), np.zeros(self.system.n_pde_rows)]
        if self.args.numerical_ridge > 0:
            n_parameters = self.observation_basis.shape[1]
            blocks.append(np.sqrt(self.args.numerical_ridge) * eye(n_parameters, format="csr"))
            target.append(np.zeros(n_parameters))
        return vstack(blocks, format="csr"), np.concatenate(target)

    def solve_for_diffusivity(self, d_nd: float, record_trace: bool = True) -> tuple[np.ndarray, float, dict[str, float | int]]:
        matrix, target = self.objective_matrix(d_nd)
        solution = lsmr(matrix, target, atol=1e-10, btol=1e-10, maxiter=self.args.lsmr_maxiter)
        coefficients = np.asarray(solution[0], dtype=np.float64)
        residual = matrix @ coefficients - target
        objective = float(np.mean(residual**2))
        diagnostic: dict[str, float | int] = {
            "d_nd": float(d_nd),
            "objective_mean_square": objective,
            "lsmr_istop": int(solution[1]),
            "lsmr_iterations": int(solution[2]),
            "lsmr_normr": float(solution[3]),
            "lsmr_conda": float(solution[6]),
        }
        if record_trace:
            self.trace.append(diagnostic)
        return coefficients, objective, diagnostic

    def fit(self) -> tuple[np.ndarray, float, dict[str, float | int]]:
        def objective(d_nd: float) -> float:
            return self.solve_for_diffusivity(float(d_nd))[1]

        optimized = minimize_scalar(
            objective,
            method="bounded",
            bounds=(self.args.d_min, self.args.d_max),
            options={"xatol": self.args.outer_xatol, "maxiter": self.args.outer_maxiter},
        )
        coefficients, final_objective, diagnostic = self.solve_for_diffusivity(float(optimized.x), record_trace=True)
        diagnostic.update(
            {
                "outer_success": int(bool(optimized.success)),
                "outer_iterations": int(optimized.nit),
                "outer_function_evaluations": int(optimized.nfev),
                "outer_message": str(optimized.message),
            }
        )
        return coefficients, float(optimized.x), diagnostic | {"objective_mean_square": final_objective}

    def predict(self, coefficients: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        basis = bilinear_control_basis(frame["lon"].to_numpy(), frame["lat"].to_numpy(), self.args.control_lat, self.args.control_lon)
        return np.asarray(basis @ coefficients, dtype=np.float64).ravel()


def protocol_payload(args: argparse.Namespace, split: SplitData, system: MapSystem) -> dict[str, object]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Static low-freedom PMIP4 PDE-MAP baseline; P2 is numerical acceptance only.",
        "stage": split.stage,
        "split": split.name,
        "run_kind": args.run_kind,
        "forcing": str(FORCING_PATH.relative_to(ROOT)).replace("\\", "/"),
        "forcing_sha256": sha256(FORCING_PATH),
        "active_domain": {"longitude_degrees": [-180.0, 180.0], "latitude_degrees": [-ACTIVE_LATITUDE_DEGREES, ACTIVE_LATITUDE_DEGREES]},
        "state_parameterization": {
            "type": "periodic bilinear control lattice with constant north/south boundary rows",
            "control_lat": args.control_lat,
            "control_lon": args.control_lon,
            "field_parameters": control_parameter_count(args.control_lat, args.control_lon),
            "additional_parameters": 1,
            "additional_parameter": "bounded global D_nd = D/(a U_ref)",
        },
        "discretization": {
            "forcing_grid_shape": [int(system.grid_lat_deg.size), int(system.grid_lon_deg.size)],
            "pde_rows": int(system.n_pde_rows),
            "longitude": "periodic centered finite difference",
            "latitude": "centered interior with second-order one-sided edge derivatives",
            "boundary_handling": "periodicity embedded in state parameterization; north/south rows are longitude-constant",
        },
        "objective": {
            "normalized_mean_squared_observation_loss_weight": args.observation_weight,
            "dimensionless_mean_squared_pde_loss_weight": args.pde_weight,
            "numerical_ridge": args.numerical_ridge,
            "d_nd_bounds": [args.d_min, args.d_max],
        },
        "optimizer": {"outer": "bounded scipy.optimize.minimize_scalar", "outer_xatol": args.outer_xatol, "outer_maxiter": args.outer_maxiter, "inner": "scipy.sparse.linalg.lsmr", "inner_maxiter": args.lsmr_maxiter},
        "normalization": {"method": "train_only_zscore_log_dep", "mean": split.mean, "std": split.std},
    }


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    split = load_p2_split() if args.stage == "p2" else load_p3_split(args.split, args.stage)
    system = load_map_system(args)
    run_id = args.run_id or f"{args.stage}_{args.split}_{args.run_kind}"
    run_dir = RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload(args, split, system)
    (run_dir / "experiment_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if args.run_kind == "dry-run":
        dry = {
            "status": "passed",
            "observation_rows": int(len(split.train)),
            "test_rows": int(len(split.test)),
            "field_parameters": int(control_parameter_count(args.control_lat, args.control_lon)),
            "pde_rows": int(system.n_pde_rows),
            "forcing_finite": True,
            "system_finite": bool(np.isfinite(system.advection_operator.data).all() and np.isfinite(system.diffusion_operator.data).all()),
        }
        (run_dir / "dry_run_summary.json").write_text(json.dumps(dry, indent=2), encoding="utf-8")
        print(json.dumps(dry, indent=2))
        return run_dir

    started = time.perf_counter()
    model = StaticPdeMap(args, split, system)
    coefficients, d_nd, diagnostic = model.fit()
    elapsed = time.perf_counter() - started
    predicted_norm = model.predict(coefficients, split.test)
    predicted_raw = predicted_norm * split.std + split.mean
    test_out = split.test.copy()
    test_out["prediction_log_dep_norm"] = predicted_norm
    test_out["prediction_log_dep"] = predicted_raw
    test_out["error_log_dep"] = predicted_raw - test_out["log_dep"]
    test_out["d_nd_constant"] = d_nd
    test_out.to_csv(run_dir / "test_predictions.csv", index=False)

    grid_lon, grid_lat = np.meshgrid(system.grid_lon_deg, system.grid_lat_deg, indexing="xy")
    grid_frame = pd.DataFrame({"lon": grid_lon.ravel(), "lat": grid_lat.ravel()})
    grid_out = grid_frame.copy()
    grid_out["prediction_log_dep_norm"] = model.predict(coefficients, grid_out)
    grid_out["prediction_log_dep"] = grid_out["prediction_log_dep_norm"] * split.std + split.mean
    grid_out["d_nd_constant"] = d_nd
    grid_out.to_csv(run_dir / "global_grid_predictions.csv", index=False)
    pd.DataFrame(model.trace).to_csv(run_dir / "diffusivity_search_trace.csv", index=False)
    np.save(run_dir / "control_coefficients.npy", coefficients)
    summary: dict[str, object] = {
        "split": split.name,
        "model": "B1_pde_map_static_lowfreedom",
        "elapsed_seconds": elapsed,
        "d_nd_constant": d_nd,
        "field_parameters": control_parameter_count(args.control_lat, args.control_lon),
        "total_fitted_parameters": control_parameter_count(args.control_lat, args.control_lon) + 1,
        "pde_rows": system.n_pde_rows,
        "prediction_finite_fraction": float(np.isfinite(predicted_raw).mean()),
        "objective_final_mean_square": diagnostic["objective_mean_square"],
        **raw_metrics(split.test["log_dep"].to_numpy(), predicted_raw),
        **diagnostic,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return run_dir


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
