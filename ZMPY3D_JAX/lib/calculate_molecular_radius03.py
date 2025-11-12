# All NumPy operations (`np.where`, boolean indexing, `np.stack`, `np.sum`, `**`, `np.sqrt`, `np.max`) have direct equivalents in `jax.numpy` and are highly suitable for JAX transformation.
# This function would be very efficient under JAX.

from typing import Sequence, Tuple, Union

import chex
import jax.numpy as jnp

from ZMPY3D_JAX.config import FLOAT_DTYPE


def _is_sparse(arr) -> bool:
    """Check if array is a JAX sparse array."""
    return hasattr(arr, "todense") or hasattr(arr, "toarray")


def calculate_molecular_radius03(
    voxel3d: Union[chex.Array, "jax.experimental.sparse.JAXSparse"],
    center: Sequence[float],
    volume_mass: float,
    default_radius_multiplier: float,
) -> Tuple[chex.Array, chex.Array]:
    """Calculates the average and maximum molecular radii from a 3D voxel density map,
    given the center of mass, total volume/mass, and a default radius multiplier.

    Args:
        voxel3d (jnp.ndarray or JAX sparse array): A 3D array or JAX sparse array representing the voxel density map.
            If sparse, will be converted to dense for computation.
        center: Center of mass coordinates.
        volume_mass: Total volume or mass of the voxelized object.
        default_radius_multiplier: Radius multiplier factor.

    Returns:
        Tuple of (average_distance, max_distance).
    """
    # Convert sparse to dense if needed
    if _is_sparse(voxel3d):
        voxel3d = voxel3d.todense()

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
