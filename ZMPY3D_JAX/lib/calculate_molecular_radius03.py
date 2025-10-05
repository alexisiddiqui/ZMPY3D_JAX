# All NumPy operations (`np.where`, boolean indexing, `np.stack`, `np.sum`, `**`, `np.sqrt`, `np.max`) have direct equivalents in `jax.numpy` and are highly suitable for JAX transformation.
# This function would be very efficient under JAX.

from typing import Sequence, Tuple

import numpy as np


def calculate_molecular_radius03(
    voxel3d: np.ndarray,
    center: Sequence[float],
    volume_mass: float,
    default_radius_multiplier: float,
) -> Tuple[float, float]:
    """Calculates the average and maximum molecular radii from a 3D voxel density map,
    given the center of mass, total volume/mass, and a default radius multiplier.

    Args:
        voxel3d (np.ndarray): A 3D NumPy array representing the voxel density map.
        center (np.ndarray): A 1D NumPy array representing the center of mass of the molecule.
        volume_mass (float): The total volume or mass of the molecule.
        default_radius_multiplier (float): A multiplier for the average radius calculation.

    Returns:
        tuple: A tuple containing:
            - average_voxel_dist2center (float): The average molecular radius.
            - max_voxel_dist2center (float): The maximum molecular radius.
    """
    has_weight = voxel3d > 0

    voxel_list = voxel3d[has_weight]

    x_coord, y_coord, z_coord = np.where(has_weight)

    voxel_list_xyz = np.stack([x_coord, y_coord, z_coord], axis=1)
    center = np.array(center)

    voxel_dist2center_squared = np.sum((voxel_list_xyz - center) ** 2, axis=1)

    average_voxel_mass2center_squared = np.sum(voxel_dist2center_squared * voxel_list) / volume_mass

    average_voxel_dist2center = (
        np.sqrt(average_voxel_mass2center_squared) * default_radius_multiplier
    )
    max_voxel_dist2center = np.sqrt(np.max(voxel_dist2center_squared))

    return average_voxel_dist2center, max_voxel_dist2center
