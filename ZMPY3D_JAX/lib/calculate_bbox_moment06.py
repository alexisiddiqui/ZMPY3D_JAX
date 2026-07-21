from functools import partial
from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp

import ZMPY3D_JAX.config as _config


@partial(jax.jit, static_argnums=(1,))
def _calculate_bbox_moment_jax(
    voxel3d: chex.Array,
    max_order: int,
    x_sample: chex.Array,
    y_sample: chex.Array,
    z_sample: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Calculate cell-integrated Cartesian moments on device."""
    powers = jnp.arange(1, max_order + 2, dtype=voxel3d.dtype)

    def cell_integrals(edges: chex.Array) -> chex.Array:
        return (
            edges[1:, jnp.newaxis] ** powers[jnp.newaxis, :]
            - edges[:-1, jnp.newaxis] ** powers[jnp.newaxis, :]
        ) / powers[jnp.newaxis, :]

    x_integrals = cell_integrals(x_sample)
    y_integrals = cell_integrals(y_sample)
    z_integrals = cell_integrals(z_sample)
    bbox_moment = jnp.einsum(
        "ia,jb,kc,ijk->abc",
        x_integrals,
        y_integrals,
        z_integrals,
        voxel3d,
        optimize="optimal",
    )
    volume_mass = bbox_moment[0, 0, 0]
    center = jnp.stack(
        (bbox_moment[1, 0, 0], bbox_moment[0, 1, 0], bbox_moment[0, 0, 1])
    ) / volume_mass
    return volume_mass, center, bbox_moment


def calculate_bbox_moment06(
    voxel3d: chex.Array, max_order: int, xyz_sample_struct: Dict[str, chex.Array]
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Calculate 3D bounding-box moments through ``max_order``.

    Args:
        voxel3d: A 3D voxel density map.
        max_order: The maximum order for calculating bounding-box moments.
        xyz_sample_struct: Cell-edge arrays under ``X_sample``, ``Y_sample``, and
            ``Z_sample``.

    Returns:
        The integrated mass, three-dimensional center of mass, and Cartesian
        moments with shape ``(max_order + 1,) * 3``.
    """
    voxel3d = jnp.asarray(voxel3d, dtype=_config.FLOAT_DTYPE)
    return _calculate_bbox_moment_jax(
        voxel3d,
        max_order,
        jnp.asarray(xyz_sample_struct["X_sample"], dtype=_config.FLOAT_DTYPE),
        jnp.asarray(xyz_sample_struct["Y_sample"], dtype=_config.FLOAT_DTYPE),
        jnp.asarray(xyz_sample_struct["Z_sample"], dtype=_config.FLOAT_DTYPE),
    )
