# Array reshaping and transposing are directly supported by `jax.numpy`.
# Element-wise complex multiplication and array indexing are JAX-compatible.
# The `np.add.at` operation is a major challenge for JAX due to immutability. It would need to be replaced by `jax.ops.index_add` or a functional equivalent.
# Conditional assignment for NaNs can be handled with `jax.numpy.where`.
# All other numerical operations are directly supported by `jax.numpy`.

from typing import Tuple

import numpy as np


def calculate_bbox_moment_2_zm05(
    max_order: int,
    g_cache_complex: np.ndarray,
    g_cache_pqr_linear: np.ndarray,
    g_cache_complex_index: np.ndarray,
    clm_cache3d: np.ndarray,
    bbox_moment: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts raw bounding box moments into Zernike moments (both raw and scaled).
    This is a core step in the Zernike moment calculation pipeline.

    Args:
        max_order (int): The maximum order of Zernike moments.
        g_cache_complex (np.ndarray): A NumPy array of complex coefficients from the G-cache.
        g_cache_pqr_linear (np.ndarray): A NumPy array of linear indices for p, q, r from the G-cache.
        g_cache_complex_index (np.ndarray): A NumPy array of indices for complex coefficients from the G-cache.
        clm_cache3d (np.ndarray): A 3D NumPy array of CLM coefficients for scaling.
        bbox_moment (np.ndarray): A 3D NumPy array of raw bounding box moments.

    Returns:
        tuple: A tuple containing:
            - z_moment_scaled (np.ndarray): A 3D NumPy array of scaled Zernike moments.
            - z_moment_raw (np.ndarray): A 3D NumPy array of raw Zernike moments.
    """

    def complex_nan():
        c = np.zeros(1, dtype=np.complex128)
        c[0] = np.nan
        return c

    max_n = max_order + 1

    bbox_moment = np.reshape(np.transpose(bbox_moment, (2, 1, 0)), -1)

    zm_geo = g_cache_complex * bbox_moment[g_cache_pqr_linear - 1]

    zm_geo_sum = np.zeros(max_n * max_n * max_n, dtype=np.complex128)

    np.add.at(zm_geo_sum, g_cache_complex_index - 1, zm_geo)

    zm_geo_sum[zm_geo_sum == 0.0] = complex_nan()

    z_moment_raw = zm_geo_sum * (3.0 / (4.0 * np.pi))
    z_moment_raw = z_moment_raw.reshape((max_n, max_n, max_n))
    z_moment_raw = np.transpose(z_moment_raw, (2, 1, 0))
    z_moment_scaled = (
        z_moment_raw * clm_cache3d
    )  # CLMCache3D is a 3D matrix, so operations are direct

    return z_moment_scaled, z_moment_raw
