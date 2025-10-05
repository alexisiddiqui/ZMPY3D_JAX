# Array creation, slicing, and `np.diff` have direct `jax.numpy` equivalents.
# `np.power` and `np.arange` are directly supported.
# `np.tensordot` is available in `jax.numpy` and is a prime candidate for JAX acceleration.
# `np.meshgrid` is available in `jax.numpy`.
# Array transposition and element-wise division are directly supported.
# This function is highly amenable to JAX transformation and would be very efficient.

from typing import Dict, Tuple

import numpy as np


def calculate_bbox_moment06(
    voxel3d: np.ndarray, max_order: int, xyz_sample_struct: Dict[str, np.ndarray]
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Calculates 3D bounding box moments up to a specified maximum order from a voxel density map.
    It uses `tensordot` for efficient computation.

    Args:
        voxel3d (np.ndarray): A 3D NumPy array representing the voxel density map.
        max_order (int): The maximum order for calculating bounding box moments.
        xyz_sample_struct (dict): A dictionary containing 'X_sample', 'Y_sample', and 'Z_sample'
                                 NumPy arrays of normalized sample coordinates.

    Returns:
        tuple: A tuple containing:
            - volume_mass (float): The total volume or mass of the voxelized object.
            - center (np.ndarray): A 1D NumPy array representing the center of mass.
            - bbox_moment (np.ndarray): A 3D NumPy array of bounding box moments.
    """
    extend_voxel3d = np.zeros(np.array(voxel3d.shape) + 1)
    extend_voxel3d[:-1, :-1, :-1] = voxel3d

    diff_extend_voxel3d = np.diff(np.diff(np.diff(extend_voxel3d, axis=0), axis=1), axis=2)

    x_power = np.power(xyz_sample_struct["X_sample"][1:, np.newaxis], np.arange(1, max_order + 2))
    y_power = np.power(xyz_sample_struct["Y_sample"][1:, np.newaxis], np.arange(1, max_order + 2))
    z_power = np.power(xyz_sample_struct["Z_sample"][1:, np.newaxis], np.arange(1, max_order + 2))

    bbox_moment = np.tensordot(
        z_power,
        np.tensordot(
            y_power, np.tensordot(x_power, diff_extend_voxel3d, axes=([0], [0])), axes=([0], [1])
        ),
        axes=([0], [2]),
    )

    p, q, r = np.meshgrid(
        np.arange(1, max_order + 2),
        np.arange(1, max_order + 2),
        np.arange(1, max_order + 2),
        indexing="ij",
    )
    bbox_moment = -np.transpose(bbox_moment, (2, 1, 0)) / p / q / r

    volume_mass = bbox_moment[0, 0, 0]
    center = [bbox_moment[1, 0, 0], bbox_moment[0, 1, 0], bbox_moment[0, 0, 1]] / volume_mass

    return volume_mass, center, bbox_moment
