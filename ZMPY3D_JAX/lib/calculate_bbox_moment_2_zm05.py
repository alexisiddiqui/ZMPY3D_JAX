from functools import partial
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp

import ZMPY3D_JAX.config as _config

from .segmented_reduction import segmented_sum_associative


class BBoxToZMCache(NamedTuple):
    """Device-resident constants used by bbox-to-Zernike conversion."""

    max_order: int
    g_coefficients: chex.Array
    pqr_indices: chex.Array
    output_indices: chex.Array
    clm: chex.Array


def prepare_bbox_to_zm_cache(
    max_order: int,
    g_cache_complex: chex.Array,
    g_cache_pqr_linear: chex.Array,
    g_cache_complex_index: chex.Array,
    clm_cache3d: chex.Array,
) -> BBoxToZMCache:
    """Materialize and normalize static bbox-to-ZM data once."""
    return BBoxToZMCache(
        max_order=int(max_order),
        g_coefficients=jnp.asarray(
            g_cache_complex, dtype=_config.COMPLEX_DTYPE
        ).reshape(-1),
        pqr_indices=jnp.asarray(g_cache_pqr_linear, dtype=jnp.int32).reshape(-1) - 1,
        output_indices=jnp.asarray(g_cache_complex_index, dtype=jnp.int32).reshape(-1) - 1,
        clm=jnp.asarray(clm_cache3d, dtype=_config.COMPLEX_DTYPE),
    )


@partial(jax.jit, static_argnums=(0, 6))
def _calculate_bbox_moment_2_zm_jax(
    max_order: int,
    g_coefficients: chex.Array,
    pqr_indices: chex.Array,
    output_indices: chex.Array,
    clm: chex.Array,
    bbox_moment: chex.Array,
    reduction_strategy: str = "auto",
) -> Tuple[chex.Array, chex.Array]:
    """Convert bbox moments using prepared device arrays."""
    max_n = max_order + 1
    bbox_flat = jnp.transpose(bbox_moment, (2, 1, 0)).reshape(-1)
    contributions = g_coefficients * bbox_flat[pqr_indices]
    if reduction_strategy not in ("auto", "scatter", "segmented_scan"):
        raise ValueError(
            "reduction_strategy must be 'auto', 'scatter', or 'segmented_scan'"
        )
    use_segmented_scan = reduction_strategy == "segmented_scan" or (
        reduction_strategy == "auto" and _config.COMPLEX_DTYPE == jnp.complex64
    )
    if use_segmented_scan:
        summed = segmented_sum_associative(
            contributions, output_indices, max_n**3
        )
    else:
        summed = jnp.zeros(max_n**3, dtype=bbox_moment.dtype)
        summed = summed.at[output_indices].add(contributions)
    nan_value = jnp.asarray(jnp.nan + 0j, dtype=bbox_moment.dtype)
    summed = jnp.where(summed == 0.0, nan_value, summed)

    z_moment_raw = summed * (3.0 / (4.0 * jnp.pi))
    z_moment_raw = jnp.transpose(z_moment_raw.reshape((max_n, max_n, max_n)), (2, 1, 0))
    z_moment_scaled = z_moment_raw * clm
    return z_moment_scaled, z_moment_raw


def calculate_bbox_moment_2_zm_cached(
    bbox_moment: chex.Array,
    cache: BBoxToZMCache,
    *,
    reduction_strategy: str = "auto",
) -> Tuple[chex.Array, chex.Array]:
    """Convert bbox moments with a reusable prepared cache."""
    return _calculate_bbox_moment_2_zm_jax(
        cache.max_order,
        cache.g_coefficients,
        cache.pqr_indices,
        cache.output_indices,
        cache.clm,
        jnp.asarray(bbox_moment, dtype=_config.COMPLEX_DTYPE),
        reduction_strategy,
    )


def calculate_bbox_moment_2_zm05(
    max_order: int,
    g_cache_complex: chex.Array,
    g_cache_pqr_linear: chex.Array,
    g_cache_complex_index: chex.Array,
    clm_cache3d: chex.Array,
    bbox_moment: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Convert bbox moments while preserving the original public signature."""
    cache = prepare_bbox_to_zm_cache(
        max_order,
        g_cache_complex,
        g_cache_pqr_linear,
        g_cache_complex_index,
        clm_cache3d,
    )
    return calculate_bbox_moment_2_zm_cached(bbox_moment, cache)
