"""Internal mixed-precision prototypes for high-order descriptor moments."""

from functools import partial

import chex
import jax
import jax.numpy as jnp

import ZMPY3D_JAX.config as _config

from .batched_descriptor import (
    _assemble_descriptor_batch,
    _calculate_3dzd_batch,
    _calculate_bbox_max_order_batch,
    _calculate_bbox_order1_batch,
    _calculate_normalization_means,
    _calculate_radius_and_samples_batch,
)
from .calculate_bbox_moment_2_zm05 import (
    BBoxToZMCache,
    _calculate_bbox_moment_2_zm_jax,
)
from .calculate_zm_by_ab_rotation01 import ZMRotationCache
from .descriptor_assembly import DescriptorAssemblyCache, DescriptorVector


PRECISION_FRONTIERS = ("cartesian_x64", "moments_x64")


def _validate_frontier(precision_frontier: str) -> None:
    if precision_frontier not in PRECISION_FRONTIERS:
        raise ValueError(f"unknown mixed-precision frontier: {precision_frontier}")
    if not jax.config.x64_enabled:
        raise RuntimeError("mixed-precision prototypes require jax_enable_x64=True")


@partial(jax.jit, static_argnums=(1, 5))
def calculate_bbox_moments_mixed_prototype(
    voxels: chex.Array,
    max_order: int,
    x_samples: chex.Array,
    y_samples: chex.Array,
    z_samples: chex.Array,
    precision_frontier: str,
) -> chex.Array:
    """Compute the complete Cartesian moment contraction in float64."""
    _validate_frontier(precision_frontier)
    return _calculate_bbox_max_order_batch(
        jnp.asarray(voxels, dtype=jnp.float64),
        max_order,
        jnp.asarray(x_samples, dtype=jnp.float64),
        jnp.asarray(y_samples, dtype=jnp.float64),
        jnp.asarray(z_samples, dtype=jnp.float64),
    )[2]


@partial(jax.jit, static_argnums=(1, 8))
def calculate_zm_mixed_prototype(
    bbox_moments: chex.Array,
    max_order: int,
    configured_g: chex.Array,
    pqr_indices: chex.Array,
    output_indices: chex.Array,
    configured_clm: chex.Array,
    x64_g: chex.Array,
    x64_clm: chex.Array,
    precision_frontier: str,
) -> tuple[chex.Array, chex.Array]:
    """Convert mixed Cartesian moments, downcasting at the selected frontier."""
    _validate_frontier(precision_frontier)
    if precision_frontier == "moments_x64":
        complex_moments = jnp.asarray(bbox_moments, dtype=jnp.complex128)
        g_coefficients = jnp.asarray(x64_g, dtype=jnp.complex128)
        clm = jnp.asarray(x64_clm, dtype=jnp.complex128)
        reduction = "segmented_scan"
    else:
        complex_moments = jnp.asarray(bbox_moments, dtype=_config.COMPLEX_DTYPE)
        g_coefficients = jnp.asarray(configured_g, dtype=_config.COMPLEX_DTYPE)
        clm = jnp.asarray(configured_clm, dtype=_config.COMPLEX_DTYPE)
        reduction = "auto"

    scaled, raw = jax.vmap(
        lambda bbox: _calculate_bbox_moment_2_zm_jax(
            max_order,
            g_coefficients,
            pqr_indices,
            output_indices,
            clm,
            bbox,
            reduction,
        )
    )(complex_moments)
    return (
        jnp.asarray(scaled, dtype=_config.COMPLEX_DTYPE),
        jnp.asarray(raw, dtype=_config.COMPLEX_DTYPE),
    )


def calculate_descriptor_mixed_prototype(
    voxels: chex.Array,
    *,
    max_order: int,
    target_orders: tuple[int, ...],
    x_samples: chex.Array,
    y_samples: chex.Array,
    z_samples: chex.Array,
    configured_bbox_cache: BBoxToZMCache,
    x64_bbox_cache: BBoxToZMCache,
    rotation_cache: ZMRotationCache,
    descriptor_cache: DescriptorAssemblyCache,
    precision_frontier: str,
) -> tuple[
    DescriptorVector,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
]:
    """Run one experimental precision frontier and expose its stage boundaries."""
    bbox = calculate_bbox_moments_mixed_prototype(
        voxels,
        max_order,
        x_samples,
        y_samples,
        z_samples,
        precision_frontier,
    )
    scaled, raw = calculate_zm_mixed_prototype(
        bbox,
        max_order,
        configured_bbox_cache.g_coefficients,
        configured_bbox_cache.pqr_indices,
        configured_bbox_cache.output_indices,
        configured_bbox_cache.clm,
        x64_bbox_cache.g_coefficients,
        x64_bbox_cache.clm,
        precision_frontier,
    )
    descriptors_3dzd = _calculate_3dzd_batch(scaled)
    means = _calculate_normalization_means(
        raw,
        target_orders,
        "companion_compact_grouped",
        rotation_cache,
        "auto",
    )
    descriptor = _assemble_descriptor_batch(
        descriptors_3dzd,
        means,
        descriptor_cache.descriptor_indices,
        descriptor_cache.moment_indices,
    )
    return descriptor, bbox, scaled, raw, descriptors_3dzd, means


def calculate_descriptor_from_voxels_mixed_prototype(
    voxels: chex.Array,
    *,
    max_order: int,
    target_orders: tuple[int, ...],
    default_radius_multiplier: float,
    configured_bbox_cache: BBoxToZMCache,
    x64_bbox_cache: BBoxToZMCache,
    rotation_cache: ZMRotationCache,
    descriptor_cache: DescriptorAssemblyCache,
    precision_frontier: str,
) -> DescriptorVector:
    """Run the complete device pipeline for one experimental frontier."""
    masses, centers, _ = _calculate_bbox_order1_batch(voxels)
    radius = _calculate_radius_and_samples_batch(
        voxels, centers, masses, default_radius_multiplier
    )
    descriptor, *_ = calculate_descriptor_mixed_prototype(
        voxels,
        max_order=max_order,
        target_orders=target_orders,
        x_samples=radius[3],
        y_samples=radius[4],
        z_samples=radius[5],
        configured_bbox_cache=configured_bbox_cache,
        x64_bbox_cache=x64_bbox_cache,
        rotation_cache=rotation_cache,
        descriptor_cache=descriptor_cache,
        precision_frontier=precision_frontier,
    )
    return descriptor
