# All operations (`np.stack`, `np.abs`, `np.mean`, `np.std`) have direct equivalents in `jax.numpy` and are highly suitable for JAX transformation.
# This function would be very efficient under JAX.

from typing import Sequence, Tuple

import numpy as np


def get_mean_invariant03(zm_list: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the mean and standard deviation of a list of Zernike moment arrays,
    typically representing different rotations of a molecule.

    Args:
        zm_list (list): A list of NumPy arrays, where each array contains Zernike moments.

    Returns:
        tuple: A tuple containing:
            - zm_mean (np.ndarray): The mean of the Zernike moments.
            - zm_std (np.ndarray): The standard deviation of the Zernike moments.
    """
    all_zm = np.abs(np.stack(zm_list, axis=3))

    zm_mean = np.mean(all_zm, axis=3)
    zm_std = np.std(all_zm, axis=3, ddof=1)

    return zm_mean, zm_std
