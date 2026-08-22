"""Run P1 mathematical and forcing-interpolation validation for v5."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

from pmip4_spherical_pde import (
    ConstantForcing,
    CoordinateSystem,
    DEFAULT_COORDINATES,
    EARTH_RADIUS_M,
    FLOAT_DTYPE,
    PMIP4VectorForcing,
    spherical_transport_terms,
    spherical_transport_terms_constd,
)


tf.disable_v2_behavior()

ROOT = Path(__file__).resolve().parents[1]
FORCING_PATH = ROOT / "Data" / "processed_data" / "pmip4_wind" / "awi_primary_3981_4000_annual_850hpa.nc"
RESULTS_DIR = ROOT / "Data" / "model_results" / "v5_pmip4_2d"


def session_run(fetches, feed_dict):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        return session.run(fetches, feed_dict=feed_dict)


def sample_points(count: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-1.8, 1.8, count), rng.uniform(-0.80, 0.80, count)]).astype(np.float64)


def test_constant_field() -> dict[str, float]:
    tf.reset_default_graph()
    x = tf.placeholder(FLOAT_DTYPE, shape=[None, 2])
    c = tf.ones_like(x[:, 0:1]) * 2.5
    lam = DEFAULT_COORDINATES.longitude(x)
    phi = DEFAULT_COORDINATES.latitude(x)
    d = 0.8 + 0.1 * tf.cos(lam) * tf.cos(phi)
    terms = spherical_transport_terms(x, c, d, ConstantForcing(4.0, 0.0), earth_radius_m=2.3)
    values = session_run(terms["total"], {x: sample_points(300)})
    error = float(np.max(np.abs(values)))
    assert error < 1e-12, error
    return {"max_abs_residual": error, "tolerance": 1e-12}


def test_constant_d_degeneration() -> dict[str, float]:
    tf.reset_default_graph()
    x = tf.placeholder(FLOAT_DTYPE, shape=[None, 2])
    lam = DEFAULT_COORDINATES.longitude(x)
    phi = DEFAULT_COORDINATES.latitude(x)
    c = tf.sin(lam) * tf.cos(phi) + 0.1 * tf.sin(2.0 * phi)
    d_value = 0.7
    forcing = ConstantForcing(2.0, -0.4)
    variable_terms = spherical_transport_terms(x, c, tf.ones_like(c) * d_value, forcing, earth_radius_m=2.3)
    constant_terms = spherical_transport_terms_constd(x, c, d_value, forcing, earth_radius_m=2.3)
    variable_value, constant_value = session_run(
        [variable_terms["total"], constant_terms["total"]], {x: sample_points(300)}
    )
    error = float(np.max(np.abs(variable_value - constant_value)))
    assert error < 1e-8, error
    return {"max_abs_difference": error, "tolerance": 1e-8}


def manufactured_expected(lambda_rad: np.ndarray, phi_rad: np.ndarray, earth_radius: float) -> np.ndarray:
    eastward, northward = 2.0, -0.4
    d0, epsilon = 0.7, 0.2
    c = np.sin(lambda_rad) * np.cos(phi_rad)
    diffusivity = d0 * (1.0 + epsilon * np.sin(lambda_rad) * np.cos(phi_rad))
    advection = -eastward / earth_radius * np.cos(lambda_rad) + 2.0 * northward / earth_radius * np.sin(lambda_rad) * np.sin(phi_rad)
    diffusion = d0 / earth_radius**2 * (
        -2.0 * (1.0 + epsilon * c) * c
        + epsilon * (np.cos(lambda_rad) ** 2 + np.sin(lambda_rad) ** 2 * np.sin(phi_rad) ** 2)
    )
    return advection + diffusion


def test_manufactured_solution() -> dict[str, float]:
    tf.reset_default_graph()
    earth_radius = 2.3
    x = tf.placeholder(FLOAT_DTYPE, shape=[None, 2])
    lam = DEFAULT_COORDINATES.longitude(x)
    phi = DEFAULT_COORDINATES.latitude(x)
    c = tf.sin(lam) * tf.cos(phi)
    d = 0.7 * (1.0 + 0.2 * tf.sin(lam) * tf.cos(phi))
    terms = spherical_transport_terms(x, c, d, ConstantForcing(2.0, -0.4), earth_radius_m=earth_radius)
    points = sample_points(700)
    actual = session_run(terms["total"], {x: points}).ravel()
    expected = manufactured_expected(points[:, 0] * np.pi / 2.0, points[:, 1] * np.pi / 2.0, earth_radius)
    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    assert error < 1e-8, error
    return {"relative_l2_error": error, "tolerance": 1e-8}


def test_coordinate_scaling() -> dict[str, float]:
    tf.reset_default_graph()
    earth_radius = 2.3
    points = sample_points(400)
    x_default = tf.placeholder(FLOAT_DTYPE, shape=[None, 2])
    lam_default = DEFAULT_COORDINATES.longitude(x_default)
    phi_default = DEFAULT_COORDINATES.latitude(x_default)
    c_default = tf.sin(lam_default) * tf.cos(phi_default)
    d_default = 0.7 * (1.0 + 0.2 * tf.sin(lam_default) * tf.cos(phi_default))
    default_terms = spherical_transport_terms(x_default, c_default, d_default, ConstantForcing(2.0, -0.4), earth_radius_m=earth_radius)

    physical_coordinates = CoordinateSystem(1.0, 1.0)
    x_physical = tf.placeholder(FLOAT_DTYPE, shape=[None, 2])
    lam_physical = physical_coordinates.longitude(x_physical)
    phi_physical = physical_coordinates.latitude(x_physical)
    c_physical = tf.sin(lam_physical) * tf.cos(phi_physical)
    d_physical = 0.7 * (1.0 + 0.2 * tf.sin(lam_physical) * tf.cos(phi_physical))
    physical_terms = spherical_transport_terms(
        x_physical,
        c_physical,
        d_physical,
        ConstantForcing(2.0, -0.4),
        coordinates=physical_coordinates,
        earth_radius_m=earth_radius,
    )
    default_value, physical_value = session_run(
        [default_terms["total"], physical_terms["total"]],
        {x_default: points, x_physical: points * np.pi / 2.0},
    )
    error = float(np.max(np.abs(default_value - physical_value)))
    assert error < 1e-12, error
    return {"max_abs_difference": error, "tolerance": 1e-12}


def finite_difference_total(lambda_grid: np.ndarray, phi_grid: np.ndarray, earth_radius: float) -> np.ndarray:
    lam, phi = np.meshgrid(lambda_grid, phi_grid, indexing="xy")
    c = np.sin(lam) * np.cos(phi)
    d = 0.7 * (1.0 + 0.2 * np.sin(lam) * np.cos(phi))
    u, v = 2.0, -0.4
    d_lambda = lambda_grid[1] - lambda_grid[0]
    d_phi = phi_grid[1] - phi_grid[0]
    derivative_lambda = lambda values: (np.roll(values, -1, axis=1) - np.roll(values, 1, axis=1)) / (2.0 * d_lambda)
    derivative_phi = lambda values: (values[2:, :] - values[:-2, :]) / (2.0 * d_phi)
    cos_phi = np.cos(phi)
    advection = -(derivative_lambda(u * c)[1:-1, :] + derivative_phi(v * c * cos_phi)) / (earth_radius * cos_phi[1:-1, :])
    dc_dlambda = derivative_lambda(c)
    dc_dphi = np.zeros_like(c)
    dc_dphi[1:-1, :] = derivative_phi(c)
    flux_lon = d * dc_dlambda / cos_phi
    flux_lat = d * dc_dphi * cos_phi
    diffusion = (derivative_lambda(flux_lon)[1:-1, :] + derivative_phi(flux_lat)) / (earth_radius**2 * cos_phi[1:-1, :])
    return advection + diffusion


def test_finite_difference_crosscheck() -> dict[str, float]:
    tf.reset_default_graph()
    earth_radius = 2.3
    lambda_grid = np.linspace(-np.pi, np.pi, 257, endpoint=False)
    phi_grid = np.linspace(-1.2, 1.2, 121)
    lam, phi = np.meshgrid(lambda_grid, phi_grid, indexing="xy")
    points = np.column_stack([(lam.ravel() / (np.pi / 2.0)), (phi.ravel() / (np.pi / 2.0))]).astype(np.float64)
    x = tf.placeholder(FLOAT_DTYPE, shape=[None, 2])
    lam_tf = DEFAULT_COORDINATES.longitude(x)
    phi_tf = DEFAULT_COORDINATES.latitude(x)
    c = tf.sin(lam_tf) * tf.cos(phi_tf)
    d = 0.7 * (1.0 + 0.2 * tf.sin(lam_tf) * tf.cos(phi_tf))
    terms = spherical_transport_terms(x, c, d, ConstantForcing(2.0, -0.4), earth_radius_m=earth_radius)
    automatic = session_run(terms["total"], {x: points}).reshape(phi_grid.size, lambda_grid.size)[2:-2, :]
    finite_difference = finite_difference_total(lambda_grid, phi_grid, earth_radius)[1:-1, :]
    error = float(np.linalg.norm(automatic - finite_difference) / np.linalg.norm(automatic))
    assert error < 0.002, error
    return {"relative_l2_error": error, "tolerance": 0.002}


def test_actual_forcing_interpolation() -> dict[str, float]:
    tf.reset_default_graph()
    forcing = PMIP4VectorForcing(FORCING_PATH)
    x = tf.placeholder(FLOAT_DTYPE, shape=[None, 2])
    eastward, northward = forcing.evaluate(x)
    lam = DEFAULT_COORDINATES.longitude(x)
    phi = DEFAULT_COORDINATES.latitude(x)
    concentration = tf.sin(lam) * tf.cos(phi)
    actual_terms = spherical_transport_terms(
        x,
        concentration,
        tf.ones_like(concentration) * 1.0e6,
        forcing,
        earth_radius_m=EARTH_RADIUS_M,
    )
    points = sample_points(1000, seed=11)
    u, v, residual = session_run([eastward, northward, actual_terms["total"]], {x: points})
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        raise AssertionError("Actual PMIP4 interpolation returned non-finite values")
    if not np.isfinite(residual).all():
        raise AssertionError("Actual PMIP4 forcing produced a non-finite differentiated PDE residual")
    seam_points = np.asarray([[-2.0, -0.5], [2.0, -0.5], [-2.0, 0.5], [2.0, 0.5]], dtype=np.float64)
    seam_u, seam_v = session_run([eastward, northward], {x: seam_points})
    seam_error = float(max(np.abs(seam_u[0] - seam_u[1]), np.abs(seam_u[2] - seam_u[3]), np.abs(seam_v[0] - seam_v[1]), np.abs(seam_v[2] - seam_v[3])))
    assert seam_error < 1e-12, seam_error
    return {
        "finite_fraction": float(np.isfinite(u).mean() * np.isfinite(v).mean()),
        "pde_residual_finite_fraction": float(np.isfinite(residual).mean()),
        "seam_max_abs_difference": seam_error,
        "tolerance": 1e-12,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tests = {
        "constant_field": test_constant_field,
        "constant_D_degeneration": test_constant_d_degeneration,
        "manufactured_solution": test_manufactured_solution,
        "coordinate_scaling": test_coordinate_scaling,
        "finite_difference_crosscheck": test_finite_difference_crosscheck,
        "actual_forcing_interpolation": test_actual_forcing_interpolation,
    }
    results: dict[str, object] = {"created_utc": datetime.now(timezone.utc).isoformat(), "forcing": str(FORCING_PATH.relative_to(ROOT)).replace("\\", "/"), "tests": {}}
    for name, function in tests.items():
        metrics = function()
        results["tests"][name] = {"status": "pass", **metrics}
        print(f"PASS {name}: {metrics}", flush=True)
    (RESULTS_DIR / "p1_pde_validation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("P1 PDE validation completed successfully.")


if __name__ == "__main__":
    main()
