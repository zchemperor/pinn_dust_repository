"""Fast numerical acceptance tests for the static PMIP4 PDE-MAP baseline."""

from __future__ import annotations

import unittest

import numpy as np

from v5_pmip4_pde_map import (
    ACTIVE_LATITUDE_DEGREES,
    CONTROL_LATITUDE_COUNT,
    CONTROL_LONGITUDE_COUNT,
    bilinear_control_basis,
    control_parameter_count,
    load_map_system,
    parse_args,
)


class StaticMapTests(unittest.TestCase):
    def test_control_dimension(self) -> None:
        self.assertEqual(control_parameter_count(12, 24), 242)

    def test_periodic_control_interpolation(self) -> None:
        basis = bilinear_control_basis(
            np.asarray([-180.0, 180.0]), np.asarray([0.0, 0.0]), CONTROL_LATITUDE_COUNT, CONTROL_LONGITUDE_COUNT
        )
        self.assertTrue(np.allclose(basis.toarray()[0], basis.toarray()[1], atol=1e-14))

    def test_polar_boundary_rows_are_longitude_constant(self) -> None:
        basis = bilinear_control_basis(
            np.asarray([-140.0, 10.0, 175.0]),
            np.asarray([-ACTIVE_LATITUDE_DEGREES, ACTIVE_LATITUDE_DEGREES, ACTIVE_LATITUDE_DEGREES]),
            CONTROL_LATITUDE_COUNT,
            CONTROL_LONGITUDE_COUNT,
        ).toarray()
        self.assertEqual(np.linalg.matrix_rank(basis[:1]), 1)
        self.assertTrue(np.allclose(basis[1], basis[2], atol=1e-14))

    def test_forcing_operator_is_finite(self) -> None:
        args = parse_args()
        system = load_map_system(args)
        self.assertEqual(system.grid_lon_deg.size, 120)
        self.assertGreater(system.grid_lat_deg.size, 40)
        self.assertEqual(system.n_pde_rows, (system.grid_lat_deg.size - 2) * system.grid_lon_deg.size)
        self.assertTrue(np.isfinite(system.advection_operator.data).all())
        self.assertTrue(np.isfinite(system.diffusion_operator.data).all())


if __name__ == "__main__":
    unittest.main()
