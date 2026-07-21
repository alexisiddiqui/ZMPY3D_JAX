# All NumPy operations (`np.where`, boolean indexing, `np.stack`, `np.sum`, `**`, `np.sqrt`, `np.max`) have direct equivalents in `jax.numpy` and are highly suitable for JAX transformation.
# This function would be very efficient under JAX.

from typing import Sequence, Tuple

import chex
import jax
import jax.numpy as jnp

import ZMPY3D_JAX.config as _config


def _radius_statistics_impl(
    voxel3d: chex.Array,
    center: Sequence[float],
    volume_mass: float,
    default_radius_multiplier: float,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    voxel3d = jnp.asarray(voxel3d, dtype=_config.FLOAT_DTYPE)
    center = jnp.asarray(center, dtype=_config.FLOAT_DTYPE)
    volume_mass = jnp.asarray(volume_mass, dtype=_config.FLOAT_DTYPE)
    default_radius_multiplier = jnp.asarray(default_radius_multiplier, dtype=_config.FLOAT_DTYPE)

    has_weight = voxel3d > 0
    x_offset = (
        jnp.arange(voxel3d.shape[0], dtype=_config.FLOAT_DTYPE) - center[0]
    )[:, None, None]
    y_offset = (
        jnp.arange(voxel3d.shape[1], dtype=_config.FLOAT_DTYPE) - center[1]
    )[None, :, None]
    z_offset = (
        jnp.arange(voxel3d.shape[2], dtype=_config.FLOAT_DTYPE) - center[2]
    )[None, None, :]
    voxel_dist2center_squared = (
        x_offset * x_offset + y_offset * y_offset + z_offset * z_offset
    )
    positive_weight = jnp.where(has_weight, voxel3d, 0)
    average_voxel_mass2center_squared = (
        jnp.sum(voxel_dist2center_squared * positive_weight) / volume_mass
    )
    average_voxel_dist2center = (
        jnp.sqrt(average_voxel_mass2center_squared) * default_radius_multiplier
    )
    max_voxel_dist2center = jnp.sqrt(
        jnp.max(jnp.where(has_weight, voxel_dist2center_squared, 0), initial=0)
    )

    return jnp.any(has_weight), average_voxel_dist2center, max_voxel_dist2center


@jax.jit
def _calculate_molecular_radius_jax(
    voxel3d: chex.Array,
    center: Sequence[float],
    volume_mass: float,
    default_radius_multiplier: float,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    return _radius_statistics_impl(
        voxel3d, center, volume_mass, default_radius_multiplier
    )


@jax.jit
def _calculate_molecular_radius_and_bbox_samples_jax(
    voxel3d: chex.Array,
    center: Sequence[float],
    volume_mass: float,
    default_radius_multiplier: float,
):
    has_weight, average_radius, max_radius = _radius_statistics_impl(
        voxel3d, center, volume_mass, default_radius_multiplier
    )
    center = jnp.asarray(center, dtype=_config.FLOAT_DTYPE)
    sphere = {
        "X_sample": (
            jnp.arange(voxel3d.shape[0] + 1, dtype=_config.FLOAT_DTYPE) - center[0]
        )
        / average_radius,
        "Y_sample": (
            jnp.arange(voxel3d.shape[1] + 1, dtype=_config.FLOAT_DTYPE) - center[1]
        )
        / average_radius,
        "Z_sample": (
            jnp.arange(voxel3d.shape[2] + 1, dtype=_config.FLOAT_DTYPE) - center[2]
        )
        / average_radius,
    }
    return has_weight, average_radius, max_radius, sphere


def _raise_for_empty_density(has_weight: chex.Array) -> None:
    if not bool(has_weight):
        raise ValueError("zero-size array to reduction operation maximum which has no identity")


def calculate_molecular_radius03(
    voxel3d: chex.Array,
    center: Sequence[float],
    volume_mass: float,
    default_radius_multiplier: float,
) -> Tuple[chex.Array, chex.Array]:
    """Calculate average and maximum molecular radii with fixed-shape reductions."""
    has_weight, average_radius, max_radius = _calculate_molecular_radius_jax(
        voxel3d, center, volume_mass, default_radius_multiplier
    )
    _raise_for_empty_density(has_weight)
    return average_radius, max_radius


def calculate_molecular_radius_and_bbox_samples(
    voxel3d: chex.Array,
    center: Sequence[float],
    volume_mass: float,
    default_radius_multiplier: float,
):
    """Calculate radii and normalized bbox samples in one compiled kernel."""
    has_weight, average_radius, max_radius, sphere = (
        _calculate_molecular_radius_and_bbox_samples_jax(
            voxel3d, center, volume_mass, default_radius_multiplier
        )
    )
    _raise_for_empty_density(has_weight)
    return average_radius, max_radius, sphere
