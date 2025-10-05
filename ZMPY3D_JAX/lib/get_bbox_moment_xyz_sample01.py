# All NumPy operations (`np.arange`, subtraction, division) have direct equivalents in `jax.numpy` and are highly suitable for JAX transformation.
# This function would be very efficient under JAX.


from typing import Dict, Tuple

import numpy as np


def get_bbox_moment_xyz_sample01(
    center: np.ndarray, radius: float, dimension_bbox_scaled: Tuple[int, int, int]
) -> Dict[str, np.ndarray]:
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
    x_edge, y_edge, z_edge = dimension_bbox_scaled

    x_sample = (np.arange(x_edge + 1) - center[0]) / radius
    y_sample = (np.arange(y_edge + 1) - center[1]) / radius
    z_sample = (np.arange(z_edge + 1) - center[2]) / radius

    xyz_sample_struct = {"X_sample": x_sample, "Y_sample": y_sample, "Z_sample": z_sample}

    return xyz_sample_struct
