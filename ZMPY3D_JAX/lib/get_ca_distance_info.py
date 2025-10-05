# All NumPy operations (`np.mean`, `np.sqrt`, `np.sum`, `**`, `np.percentile`, `np.std`) have direct equivalents in `jax.numpy` and are highly suitable for JAX transformation.
# This function would benefit significantly from JAX's numerical acceleration and automatic differentiation capabilities.

from typing import Tuple

import numpy as np


def get_ca_distance_info(xyz: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """Calculates various geometric descriptors from C-alpha (CA) atom coordinates,
    including percentiles of distances to the center, standard deviation of distances,
    skewness (s), and kurtosis (k).

    Args:
        xyz (np.ndarray): A NumPy array of shape (N, 3) with C-alpha atom coordinates.

    Returns:
        tuple: A tuple containing:
            - percentile_list (np.ndarray): Percentiles of distances to the center.
            - std_xyz_dist2center (float): Standard deviation of distances to the center.
            - s (float): Skewness of the distances.
            - k (float): Kurtosis of the distances.
    """
    xyz_center = np.mean(xyz, axis=0)
    xyz_dist2center = np.sqrt(np.sum((xyz - xyz_center) ** 2, axis=1))

    percentiles_for_geom = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    percentile_list = np.percentile(xyz_dist2center, percentiles_for_geom)
    percentile_list = percentile_list.reshape(-1, 1)

    std_xyz_dist2center = np.std(xyz_dist2center, ddof=1)

    n = len(xyz_dist2center)
    mean_distance = np.mean(xyz_dist2center)
    s = (n / ((n - 1) * (n - 2))) * np.sum(
        ((xyz_dist2center - mean_distance) / std_xyz_dist2center) ** 3
    )

    fourth_moment = np.sum(((xyz_dist2center - mean_distance) / std_xyz_dist2center) ** 4)
    k = n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * fourth_moment - 3 * (n - 1) ** 2 / (
        (n - 2) * (n - 3)
    )

    return percentile_list, std_xyz_dist2center, s, k
