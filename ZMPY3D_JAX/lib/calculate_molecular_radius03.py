# All NumPy operations (`np.where`, boolean indexing, `np.stack`, `np.sum`, `**`, `np.sqrt`, `np.max`) have direct equivalents in `jax.numpy` and are highly suitable for JAX transformation.
# This function would be very efficient under JAX.

from typing import Sequence, Tuple

import chex
import jax.numpy as jnp

from ZMPY3D_JAX.config import FLOAT_DTYPE


def calculate_molecular_radius03(
    voxel3d: chex.Array,
    center: Sequence[float],
    volume_mass: float,
    default_radius_multiplier: float,
) -> Tuple[chex.Array, chex.Array]:
    """Calculates the average and maximum molecular radii from a 3D voxel density map,
    given the center of mass, total volume/mass, and a default radius multiplier.
    """
    voxel3d = jnp.asarray(voxel3d, dtype=FLOAT_DTYPE)
    center = jnp.asarray(center, dtype=FLOAT_DTYPE)
    volume_mass = jnp.asarray(volume_mass, dtype=FLOAT_DTYPE)
    default_radius_multiplier = jnp.asarray(default_radius_multiplier, dtype=FLOAT_DTYPE)

    has_weight = voxel3d > 0

    voxel_list = voxel3d[has_weight]

    x_coord, y_coord, z_coord = jnp.where(has_weight)

    voxel_list_xyz = jnp.stack([x_coord, y_coord, z_coord], axis=1).astype(FLOAT_DTYPE)

    voxel_dist2center_squared = jnp.sum((voxel_list_xyz - center) ** 2, axis=1)

    average_voxel_mass2center_squared = (
        jnp.sum(voxel_dist2center_squared * voxel_list) / volume_mass
    )

    average_voxel_dist2center = (
        jnp.sqrt(average_voxel_mass2center_squared) * default_radius_multiplier
    )
    max_voxel_dist2center = jnp.sqrt(jnp.max(voxel_dist2center_squared))

    return average_voxel_dist2center, max_voxel_dist2center
