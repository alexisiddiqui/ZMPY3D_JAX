# All numerical operations (`np.abs`, `**`, `np.sum`, `np.sqrt`) have direct equivalents in `jax.numpy`.
# Array slicing and conditional assignments can be handled with `jax.numpy.where` or `jax.lax.select`.
# This function is highly suitable for JAX transformation.


import numpy as np


def get_3dzd_121_descriptor02(z_moment_scaled: np.ndarray) -> np.ndarray:
    """Calculates the 3D Zernike Descriptor (3DZD) 121 invariant from scaled Zernike moments.
    This invariant is a rotation-invariant descriptor of molecular shape.

    Args:
        z_moment_scaled (np.ndarray): A NumPy array of scaled Zernike moments.

    Returns:
        np.ndarray: A NumPy array representing the 3DZD 121 invariant.
    """
    z_moment_scaled[np.isnan(z_moment_scaled)] = 0
    z_moment_scaled_norm = np.abs(z_moment_scaled) ** 2

    z_moment_scaled_norm_positive = np.sum(z_moment_scaled_norm, axis=2)

    z_moment_scaled_norm[:, :, 0] = 0
    z_moment_scaled_norm_negative = np.sum(z_moment_scaled_norm, axis=2)

    zm_3dzd_invariant = np.sqrt(z_moment_scaled_norm_positive + z_moment_scaled_norm_negative)
    zm_3dzd_invariant[zm_3dzd_invariant < 1e-20] = np.nan

    return zm_3dzd_invariant
