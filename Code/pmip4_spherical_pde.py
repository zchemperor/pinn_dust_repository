"""Differentiable PMIP4 vector-wind interpolation and spherical PDE terms.

Inputs use the project's normalized coordinates: x[:, 0] = longitude / 90 and
x[:, 1] = latitude / 90. The helpers below recover derivatives in radians
before evaluating the conservative spherical transport residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf
from netCDF4 import Dataset


tf.disable_v2_behavior()

EARTH_RADIUS_M = 6_371_000.0
FLOAT_DTYPE = tf.float64


@dataclass(frozen=True)
class CoordinateSystem:
    """Map network coordinates to longitude and latitude in radians."""

    lon_radians_per_unit: float = np.pi / 2.0
    lat_radians_per_unit: float = np.pi / 2.0

    def longitude(self, x: tf.Tensor) -> tf.Tensor:
        return tf.cast(x[:, 0:1], FLOAT_DTYPE) * self.lon_radians_per_unit

    def latitude(self, x: tf.Tensor) -> tf.Tensor:
        return tf.cast(x[:, 1:2], FLOAT_DTYPE) * self.lat_radians_per_unit

    def longitude_derivative(self, derivative_wrt_x: tf.Tensor) -> tf.Tensor:
        return tf.cast(derivative_wrt_x, FLOAT_DTYPE) / self.lon_radians_per_unit

    def latitude_derivative(self, derivative_wrt_y: tf.Tensor) -> tf.Tensor:
        return tf.cast(derivative_wrt_y, FLOAT_DTYPE) / self.lat_radians_per_unit


DEFAULT_COORDINATES = CoordinateSystem()


class PMIP4VectorForcing:
    """Piecewise-bilinear, periodic `U,V` interpolation on a regular grid."""

    def __init__(self, netcdf_path: str | Path, coordinates: CoordinateSystem = DEFAULT_COORDINATES):
        self.path = Path(netcdf_path)
        self.coordinates = coordinates
        with Dataset(self.path) as dataset:
            lon_deg = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
            lat_deg = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
            ua = np.asarray(dataset.variables["ua_annual"][:], dtype=np.float64)
            va = np.asarray(dataset.variables["va_annual"][:], dtype=np.float64)
        if ua.shape != (lat_deg.size, lon_deg.size) or va.shape != ua.shape:
            raise ValueError(f"{self.path.name}: inconsistent forcing dimensions")
        if not (np.isfinite(ua).all() and np.isfinite(va).all()):
            raise ValueError(f"{self.path.name}: forcing contains non-finite values")
        if not np.all(np.diff(lat_deg) > 0):
            raise ValueError(f"{self.path.name}: latitude must be strictly increasing")

        lon_rad = np.deg2rad(lon_deg)
        lon_periodic = np.mod(lon_rad, 2.0 * np.pi)
        order = np.argsort(lon_periodic)
        lon_periodic = lon_periodic[order]
        ua = ua[:, order]
        va = va[:, order]
        keep = np.concatenate([[True], np.diff(lon_periodic) > 1e-12])
        lon_periodic = lon_periodic[keep]
        ua = ua[:, keep]
        va = va[:, keep]
        if lon_periodic.size < 4:
            raise ValueError(f"{self.path.name}: insufficient longitude resolution")

        self.lat = tf.constant(np.deg2rad(lat_deg), dtype=FLOAT_DTYPE)
        self.lon = tf.constant(lon_periodic, dtype=FLOAT_DTYPE)
        self.lon_extended = tf.concat([self.lon, self.lon[:1] + 2.0 * np.pi], axis=0)
        self.ua = tf.constant(ua, dtype=FLOAT_DTYPE)
        self.va = tf.constant(va, dtype=FLOAT_DTYPE)

    def _interpolate(self, field: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        longitude = self.coordinates.longitude(x)
        latitude = self.coordinates.latitude(x)
        lon0 = self.lon[0]
        wrapped_lon = tf.where(longitude < lon0, longitude + 2.0 * np.pi, longitude)
        wrapped_lon = tf.where(wrapped_lon >= lon0 + 2.0 * np.pi, wrapped_lon - 2.0 * np.pi, wrapped_lon)
        clipped_lat = tf.clip_by_value(latitude, self.lat[0], self.lat[-1])
        flat_lon = tf.reshape(wrapped_lon, [-1])
        flat_lat = tf.reshape(clipped_lat, [-1])

        lat_index = tf.searchsorted(self.lat, flat_lat, side="right") - 1
        lon_index = tf.searchsorted(self.lon_extended, flat_lon, side="right") - 1
        lat_index = tf.clip_by_value(lat_index, 0, tf.shape(self.lat)[0] - 2)
        lon_index = tf.clip_by_value(lon_index, 0, tf.shape(self.lon)[0] - 1)
        lat_next = lat_index + 1
        lon_next = lon_index + 1

        field_extended = tf.concat([field, field[:, :1]], axis=1)
        indices_00 = tf.stack([lat_index, lon_index], axis=1)
        indices_01 = tf.stack([lat_index, lon_next], axis=1)
        indices_10 = tf.stack([lat_next, lon_index], axis=1)
        indices_11 = tf.stack([lat_next, lon_next], axis=1)
        f00 = tf.gather_nd(field_extended, indices_00)
        f01 = tf.gather_nd(field_extended, indices_01)
        f10 = tf.gather_nd(field_extended, indices_10)
        f11 = tf.gather_nd(field_extended, indices_11)

        lat0 = tf.gather(self.lat, lat_index)
        lat1 = tf.gather(self.lat, lat_next)
        lon0_values = tf.gather(self.lon_extended, lon_index)
        lon1_values = tf.gather(self.lon_extended, lon_next)
        lat_weight = (flat_lat - lat0) / (lat1 - lat0)
        lon_weight = (flat_lon - lon0_values) / (lon1_values - lon0_values)
        lower = f00 + lon_weight * (f01 - f00)
        upper = f10 + lon_weight * (f11 - f10)
        return tf.reshape(lower + lat_weight * (upper - lower), [-1, 1])

    def evaluate(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return self._interpolate(self.ua, x), self._interpolate(self.va, x)


class ConstantForcing:
    """Analytic forcing used by PDE tests."""

    def __init__(self, eastward: float, northward: float):
        self.eastward = float(eastward)
        self.northward = float(northward)

    def evaluate(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        shape = tf.stack([tf.shape(x)[0], 1])
        return (
            tf.fill(shape, tf.cast(self.eastward, FLOAT_DTYPE)),
            tf.fill(shape, tf.cast(self.northward, FLOAT_DTYPE)),
        )


class ForcingMode:
    """Expose zero, zonal-only, or complete vector forcing for diagnostics."""

    def __init__(self, base_forcing, mode: str):
        if mode not in {"zero", "u_only", "uv"}:
            raise ValueError(f"Unknown forcing mode: {mode}")
        self.base_forcing = base_forcing
        self.mode = mode

    def evaluate(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        eastward, northward = self.base_forcing.evaluate(x)
        if self.mode == "zero":
            return tf.zeros_like(eastward), tf.zeros_like(northward)
        if self.mode == "u_only":
            return eastward, tf.zeros_like(northward)
        return eastward, northward


def _gradient(value: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    gradient = tf.gradients(value, x, unconnected_gradients="zero")[0]
    return tf.cast(gradient, FLOAT_DTYPE)


def spherical_transport_terms(
    x: tf.Tensor,
    concentration: tf.Tensor,
    diffusivity: tf.Tensor,
    forcing,
    coordinates: CoordinateSystem = DEFAULT_COORDINATES,
    earth_radius_m: float = EARTH_RADIUS_M,
) -> dict[str, tf.Tensor]:
    """Return conservative spherical advection-diffusion terms.

    The returned total is `advection + diffusion`. A source/sink term is not
    included in v5's minimal PDE and must be supplied explicitly by a later
    model if that extension is approved.
    """

    x = tf.cast(x, FLOAT_DTYPE)
    concentration = tf.cast(concentration, FLOAT_DTYPE)
    diffusivity = tf.cast(diffusivity, FLOAT_DTYPE)
    phi = coordinates.latitude(x)
    cos_phi = tf.cos(phi)
    if earth_radius_m <= 0:
        raise ValueError("earth_radius_m must be positive")

    eastward, northward = forcing.evaluate(x)
    advective_flux_lon = eastward * concentration
    advective_flux_lat = northward * concentration * cos_phi
    advection = -(
        coordinates.longitude_derivative(_gradient(advective_flux_lon, x)[:, 0:1])
        + coordinates.latitude_derivative(_gradient(advective_flux_lat, x)[:, 1:2])
    ) / (earth_radius_m * cos_phi)

    concentration_gradient = _gradient(concentration, x)
    dc_dlon = coordinates.longitude_derivative(concentration_gradient[:, 0:1])
    dc_dlat = coordinates.latitude_derivative(concentration_gradient[:, 1:2])
    diffusive_flux_lon = diffusivity * dc_dlon / cos_phi
    diffusive_flux_lat = diffusivity * dc_dlat * cos_phi
    diffusion = (
        coordinates.longitude_derivative(_gradient(diffusive_flux_lon, x)[:, 0:1])
        + coordinates.latitude_derivative(_gradient(diffusive_flux_lat, x)[:, 1:2])
    ) / (earth_radius_m**2 * cos_phi)
    return {
        "eastward_wind": eastward,
        "northward_wind": northward,
        "advection": advection,
        "diffusion": diffusion,
        "total": advection + diffusion,
    }


def spherical_transport_terms_constd(
    x: tf.Tensor,
    concentration: tf.Tensor,
    diffusivity: float,
    forcing,
    coordinates: CoordinateSystem = DEFAULT_COORDINATES,
    earth_radius_m: float = EARTH_RADIUS_M,
) -> dict[str, tf.Tensor]:
    """Independent constant-D form used to verify variable-D degeneration."""

    x = tf.cast(x, FLOAT_DTYPE)
    concentration = tf.cast(concentration, FLOAT_DTYPE)
    phi = coordinates.latitude(x)
    cos_phi = tf.cos(phi)
    eastward, northward = forcing.evaluate(x)
    advective_flux_lon = eastward * concentration
    advective_flux_lat = northward * concentration * cos_phi
    advection = -(
        coordinates.longitude_derivative(_gradient(advective_flux_lon, x)[:, 0:1])
        + coordinates.latitude_derivative(_gradient(advective_flux_lat, x)[:, 1:2])
    ) / (earth_radius_m * cos_phi)

    first_gradient = _gradient(concentration, x)
    dc_dlon = coordinates.longitude_derivative(first_gradient[:, 0:1])
    dc_dlat = coordinates.latitude_derivative(first_gradient[:, 1:2])
    second_lon = coordinates.longitude_derivative(_gradient(dc_dlon, x)[:, 0:1])
    second_lat = coordinates.latitude_derivative(_gradient(dc_dlat, x)[:, 1:2])
    laplacian = second_lon / cos_phi**2 + second_lat - tf.tan(phi) * dc_dlat
    diffusion = tf.cast(diffusivity, FLOAT_DTYPE) * laplacian / earth_radius_m**2
    return {"advection": advection, "diffusion": diffusion, "total": advection + diffusion}
