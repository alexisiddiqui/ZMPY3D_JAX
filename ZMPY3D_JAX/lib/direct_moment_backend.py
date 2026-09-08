"""Experimental float32 direct-recurrence Zernike moment backend.

The backend integrates the same piecewise-constant voxel cells as the legacy
Cartesian route.  Tensor-product Gauss--Legendre quadrature is exact for the
polynomial basis when ``2*q - 1 >= max_order``; no unit-ball clipping is done.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

import ZMPY3D_JAX.config as _config
from .zernike_basis_recurrence import basis, mode_indices, pairwise_sum


class DirectMomentCache(NamedTuple):
    max_order: int
    mode_indices: chex.Array
    convention_factors: chex.Array
    clm: chex.Array
    quadrature_offsets: chex.Array
    quadrature_weights: chex.Array


class PackedOccupiedVoxels(NamedTuple):
    """Host-compacted cells, padded only to the largest occupancy in a batch."""
    coordinates: chex.Array
    weights: chex.Array


def prepare_direct_moment_cache(max_order: int, clm, *, quadrature_order: int | None = None) -> DirectMomentCache:
    """Prepare closed-form convention factors and exact cell quadrature."""
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    q = (max_order + 2) // 2 if quadrature_order is None else int(quadrature_order)
    if q < 1:
        raise ValueError("quadrature_order must be positive")
    dtype = np.dtype(_config.FLOAT_DTYPE)
    complex_dtype = np.dtype(_config.COMPLEX_DTYPE)
    nodes, weights = np.polynomial.legendre.leggauss(q)
    grid = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
    weight_grid = np.prod(np.stack(np.meshgrid(weights, weights, weights, indexing="ij"), axis=-1), axis=-1).reshape(-1)
    indices = mode_indices(max_order)
    clm_np = np.asarray(clm, dtype=complex_dtype)
    factors = np.asarray(
        [math.sqrt(4.0 * math.pi / 3.0) * (1j ** int(m)) / clm_np[n, ell, m] for n, ell, m in indices],
        dtype=complex_dtype,
    )
    return DirectMomentCache(
        int(max_order), jnp.asarray(indices, dtype=jnp.int32), jnp.asarray(factors),
        jnp.asarray(clm_np), jnp.asarray(grid, dtype=dtype), jnp.asarray(weight_grid, dtype=dtype),
    )


def pack_occupied_voxels(voxels) -> PackedOccupiedVoxels:
    """Compact positive cells on the host and use finite origin padding."""
    data = np.asarray(voxels, dtype=np.dtype(_config.FLOAT_DTYPE))
    if data.ndim != 4 or data.shape[0] == 0:
        raise ValueError("voxels must have shape (batch, x, y, z)")
    occupied = [np.argwhere(item != 0) for item in data]
    width = max((len(item) for item in occupied), default=0)
    coordinates = np.zeros((len(data), width, 3), dtype=np.dtype(_config.FLOAT_DTYPE))
    weights = np.zeros((len(data), width), dtype=np.dtype(_config.FLOAT_DTYPE))
    for batch, cells in enumerate(occupied):
        coordinates[batch, : len(cells)] = cells
        weights[batch, : len(cells)] = data[batch][tuple(cells.T)]
    return PackedOccupiedVoxels(jnp.asarray(coordinates), jnp.asarray(weights))


def _dense_device_cells(voxels) -> PackedOccupiedVoxels:
    """Static-shape fallback used when the whole descriptor call is traced."""
    voxels = jnp.asarray(voxels)
    coordinates = jnp.stack(
        jnp.meshgrid(
            jnp.arange(voxels.shape[1]),
            jnp.arange(voxels.shape[2]),
            jnp.arange(voxels.shape[3]),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    return PackedOccupiedVoxels(
        jnp.broadcast_to(coordinates, (voxels.shape[0], *coordinates.shape)),
        voxels.reshape(voxels.shape[0], -1),
    )


def _one_direct(cell_coordinates, cell_weights, x_edges, y_edges, z_edges, cache, tile_size):
    dtype = cell_weights.dtype
    widths = jnp.asarray([x_edges[1] - x_edges[0], y_edges[1] - y_edges[0], z_edges[1] - z_edges[0]], dtype=dtype)
    origins = jnp.asarray([x_edges[0], y_edges[0], z_edges[0]], dtype=dtype)
    centers = origins + (cell_coordinates + 0.5) * widths
    offsets = cache.quadrature_offsets * (widths / 2.0)
    qweights = cache.quadrature_weights * (jnp.prod(widths) / 8.0)
    count = centers.shape[0]
    tile = count if tile_size is None else min(int(tile_size), count)
    tile = max(tile, 1)
    padded = ((count + tile - 1) // tile) * tile
    centers = jnp.pad(centers, ((0, padded - count), (0, 0)))
    cell_weights = jnp.pad(cell_weights, (0, padded - count))
    contributions = []
    for start in range(0, padded, tile):
        points = (centers[start:start + tile, None, :] + offsets[None, :, :]).reshape(-1, 3)
        _, values = basis(points, cache.max_order, dtype=dtype)
        weights = (cell_weights[start:start + tile, None] * qweights[None, :]).reshape(-1)
        contributions.append(pairwise_sum(values * weights[:, None], axis=0))
    coefficients = pairwise_sum(jnp.stack(contributions), axis=0) * cache.convention_factors
    raw_values = coefficients * (3.0 / (4.0 * jnp.pi))
    shape = (cache.max_order + 1,) * 3
    nan = jnp.asarray(jnp.nan + 0j, dtype=raw_values.dtype)
    raw = jnp.full(shape, nan, dtype=raw_values.dtype).at[
        cache.mode_indices[:, 0], cache.mode_indices[:, 1], cache.mode_indices[:, 2]
    ].set(raw_values)
    return raw * cache.clm, raw


def calculate_direct_moments(voxels_or_packed, x_edges, y_edges, z_edges, cache: DirectMomentCache, *, tile_size: int | None = 256):
    """Calculate batched ``(scaled, raw)`` moments from dense or packed cells."""
    if cache.max_order < 0:
        raise ValueError("invalid direct moment cache")
    if isinstance(voxels_or_packed, PackedOccupiedVoxels):
        packed = voxels_or_packed
    elif isinstance(voxels_or_packed, jax.core.Tracer):
        packed = _dense_device_cells(voxels_or_packed)
    else:
        packed = pack_occupied_voxels(voxels_or_packed)
    return jax.vmap(lambda c, w, x, y, z: _one_direct(c, w, x, y, z, cache, tile_size))(
        packed.coordinates, packed.weights, x_edges, y_edges, z_edges
    )
