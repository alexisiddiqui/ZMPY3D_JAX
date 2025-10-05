# Array creation, slicing, and `jnp.diff` have direct `jax.numpy` equivalents.
# `jnp.power` and `jnp.arange` are directly supported.
# `jnp.tensordot` is available in `jax.numpy` and is a prime candidate for JAX acceleration.
# `jnp.meshgrid` is available in `jax.numpy`.
# Array transposition and element-wise division are directly supported.
# This function is highly amenable to JAX transformation and would be very efficient.

from typing import Dict, Tuple

import chex
import jax.numpy as jnp

from ZMPY3D_JAX.config import FLOAT_DTYPE


def calculate_bbox_moment06(
    voxel3d: chex.Array, max_order: int, xyz_sample_struct: Dict[str, chex.Array]
) -> Tuple[float, chex.Array, chex.Array]:
    """Calculates 3D bounding box moments up to a specified maximum order from a voxel density map.
    It uses `tensordot` for efficient computation.

    Args:
        voxel3d (jnp.ndarray): A 3D NumPy array representing the voxel density map.
        max_order (int): The maximum order for calculating bounding box moments.
        xyz_sample_struct (dict): A dictionary containing 'X_sample', 'Y_sample', and 'Z_sample'
            NumPy arrays of normalized sample coordinates.

    Returns:
        tuple: A tuple containing:
            - volume_mass (float): The total volume or mass of the voxelized object.
            - center (jnp.ndarray): A 1D NumPy array representing the center of mass.
            - bbox_moment (jnp.ndarray): A 3D NumPy array of bounding box moments.
    """
    voxel3d = jnp.asarray(voxel3d, dtype=FLOAT_DTYPE)
    extend_voxel3d = jnp.zeros(jnp.array(voxel3d.shape) + 1)

    # JAX immutable update: use .at[] syntax instead of in-place assignment
    extend_voxel3d = extend_voxel3d.at[:-1, :-1, :-1].set(voxel3d)

    diff_extend_voxel3d = jnp.diff(jnp.diff(jnp.diff(extend_voxel3d, axis=0), axis=1), axis=2)

    x_power = jnp.power(
        xyz_sample_struct["X_sample"][1:, jnp.newaxis], jnp.arange(1, max_order + 2)
    )
    y_power = jnp.power(
        xyz_sample_struct["Y_sample"][1:, jnp.newaxis], jnp.arange(1, max_order + 2)
    )
    z_power = jnp.power(
        xyz_sample_struct["Z_sample"][1:, jnp.newaxis], jnp.arange(1, max_order + 2)
    )

    bbox_moment = jnp.tensordot(
        z_power,
        jnp.tensordot(
            y_power, jnp.tensordot(x_power, diff_extend_voxel3d, axes=([0], [0])), axes=([0], [1])
        ),
        axes=([0], [2]),
    )

    p, q, r = jnp.meshgrid(
        jnp.arange(1, max_order + 2),
        jnp.arange(1, max_order + 2),
        jnp.arange(1, max_order + 2),
        indexing="ij",
    )

    bbox_moment = -jnp.transpose(bbox_moment, (2, 1, 0)) / p / q / r
    volume_mass = bbox_moment[0, 0, 0]
    center = (
        jnp.array([bbox_moment[1, 0, 0], bbox_moment[0, 1, 0], bbox_moment[0, 0, 1]]) / volume_mass
    )

    return volume_mass, center, bbox_moment
