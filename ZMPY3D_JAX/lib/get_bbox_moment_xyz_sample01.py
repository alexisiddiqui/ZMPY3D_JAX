# All NumPy operations (`np.arange`, subtraction, division) have direct equivalents in `jax.numpy` and are highly suitable for JAX transformation.
# This function would be very efficient under JAX.


from typing import Dict, Tuple

import chex
import jax.numpy as jnp

from ZMPY3D_JAX.config import FLOAT_DTYPE


def get_bbox_moment_xyz_sample01(
    center: chex.Array, radius: chex.Array, dimension_bbox_scaled: Tuple[int, int, int]
) -> Dict[str, chex.Array]:
    """Generates normalized sample coordinates (X, Y, Z) for a bounding box,
    centered at a given point and scaled by a radius. These samples are used
    for calculating Zernike moments.

    Args:
        center (np.ndarray): A 1D NumPy array representing the center of the bounding box.
        radius (float): The scaling radius for the coordinates.
        dimension_bbox_scaled (tuple): A tuple (x_edge, y_edge, z_edge) representing the
                                       scaled dimensions of the bounding box.

    Returns:
        dict: A dictionary containing 'X_sample', 'Y_sample', and 'Z_sample' NumPy arrays
              of the normalized sample coordinates.
    """
    center = jnp.asarray(center, dtype=FLOAT_DTYPE)
    radius = jnp.asarray(radius, dtype=FLOAT_DTYPE)

    x_edge, y_edge, z_edge = dimension_bbox_scaled

    x_sample = (jnp.arange(x_edge + 1, dtype=FLOAT_DTYPE) - center[0]) / radius
    y_sample = (jnp.arange(y_edge + 1, dtype=FLOAT_DTYPE) - center[1]) / radius
    z_sample = (jnp.arange(z_edge + 1, dtype=FLOAT_DTYPE) - center[2]) / radius

    xyz_sample_struct = {"X_sample": x_sample, "Y_sample": y_sample, "Z_sample": z_sample}

    return xyz_sample_struct
