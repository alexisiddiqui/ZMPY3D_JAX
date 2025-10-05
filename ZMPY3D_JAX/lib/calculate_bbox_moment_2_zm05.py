# Array reshaping and transposing are directly supported by `jax.numpy`.
# Element-wise complex multiplication and array indexing are JAX-compatible.
# The `np.add.at` operation is replaced with JAX's .at[].add() for immutable updates.
# Conditional assignment for NaNs is handled with `jax.numpy.where`.
# All numerical operations are directly supported by `jax.numpy`.

from typing import Tuple

import chex
import jax.numpy as jnp

from ZMPY3D_JAX.config import COMPLEX_DTYPE


def calculate_bbox_moment_2_zm05(
    max_order: int,
    g_cache_complex: chex.Array,
    g_cache_pqr_linear: chex.Array,
    g_cache_complex_index: chex.Array,
    clm_cache3d: chex.Array,
    bbox_moment: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Converts raw bounding box moments into Zernike moments (both raw and scaled).
    This is a core step in the Zernike moment calculation pipeline.

    Args:
        max_order (int): The maximum order of Zernike moments.
        g_cache_complex (chex.Array): A JAX array of complex coefficients from the G-cache.
        g_cache_pqr_linear (chex.Array): A JAX array of linear indices for p, q, r from the G-cache.
        g_cache_complex_index (chex.Array): A JAX array of indices for complex coefficients from the G-cache.
        clm_cache3d (chex.Array): A 3D JAX array of CLM coefficients for scaling.
        bbox_moment (chex.Array): A 3D JAX array of raw bounding box moments.

    Returns:
        tuple: A tuple containing:
            - z_moment_scaled (chex.Array): A 3D JAX array of scaled Zernike moments.
            - z_moment_raw (chex.Array): A 3D JAX array of raw Zernike moments.
    """

    g_cache_complex = jnp.asarray(g_cache_complex, dtype=COMPLEX_DTYPE)
    g_cache_pqr_linear = jnp.asarray(g_cache_pqr_linear, dtype=jnp.int32)
    g_cache_complex_index = jnp.asarray(g_cache_complex_index, dtype=jnp.int32)
    clm_cache3d = jnp.asarray(clm_cache3d, dtype=COMPLEX_DTYPE)
    bbox_moment = jnp.asarray(bbox_moment, dtype=COMPLEX_DTYPE)

    max_n = max_order + 1

    bbox_moment = jnp.reshape(jnp.transpose(bbox_moment, (2, 1, 0)), -1)

    zm_geo = g_cache_complex * bbox_moment[g_cache_pqr_linear - 1]

    zm_geo_sum = jnp.zeros(max_n * max_n * max_n, dtype=COMPLEX_DTYPE)

    # JAX immutable update: use .at[].add() instead of np.add.at
    zm_geo_sum = zm_geo_sum.at[g_cache_complex_index - 1].add(zm_geo)

    # Use jnp.where for conditional NaN assignment
    zm_geo_sum = jnp.where(zm_geo_sum == 0.0, jnp.nan + 0j, zm_geo_sum)

    z_moment_raw = zm_geo_sum * (3.0 / (4.0 * jnp.pi))
    z_moment_raw = z_moment_raw.reshape((max_n, max_n, max_n))
    z_moment_raw = jnp.transpose(z_moment_raw, (2, 1, 0))
    z_moment_scaled = (
        z_moment_raw * clm_cache3d
    )  # CLMCache3D is a 3D matrix, so operations are direct

    return z_moment_scaled, z_moment_raw
