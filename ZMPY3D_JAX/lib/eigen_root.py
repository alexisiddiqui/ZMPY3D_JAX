# All NumPy operations (`np.reshape`, `np.diag`, array assignment, `np.linalg.eigvals`) have direct equivalents in `jax.numpy` and `jax.numpy.linalg`.
# This function is highly suitable for JAX transformation and would benefit from JAX's optimized linear algebra routines.


from typing import Sequence

import numpy as np


def eigen_root(poly_coefficient_list: Sequence[complex]) -> np.ndarray:
    """Calculates the roots of a polynomial given its coefficients by constructing
    a companion matrix and finding its eigenvalues.

    Args:
        poly_coefficient_list (list or np.ndarray): A list or NumPy array of polynomial coefficients.
                                                    The coefficients should be ordered from highest to lowest degree.

    Returns:
        np.ndarray: A NumPy array containing the roots of the polynomial.
    """
    poly_coefficient_list = np.reshape(poly_coefficient_list, -1)
    n = len(poly_coefficient_list) - 1

    m = np.diag(np.ones(n - 1, dtype=np.complex128), -1)
    m[0, :] = -poly_coefficient_list[1:] / poly_coefficient_list[0]
    result = np.linalg.eigvals(m)

    return result
