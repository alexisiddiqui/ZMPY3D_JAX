# This function is highly amenable to JAX conversion.
# All NumPy operations involving complex numbers, array reshaping, and linear algebra can be directly replaced with their `jax.numpy` equivalents for potential performance gains and automatic differentiation.


import numpy as np


def get_transform_matrix_from_ab_list02(
    a: complex, b: complex, center_scaled: np.ndarray
) -> np.ndarray:
    """Constructs a 4x4 transformation matrix (including rotation and translation)
    from complex 'a' and 'b' coefficients and a 'center_scaled' vector.
    This matrix is used for superimposition.

    Args:
        a (complex): The complex 'a' coefficient.
        b (complex): The complex 'b' coefficient.
        center_scaled (np.ndarray): A 1D NumPy array representing the scaled center of the molecule.

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    a2pb2 = a**2 + b**2
    a2mb2 = a**2 - b**2

    m33_linear = np.array(
        [
            np.real(a2pb2),
            -np.imag(a2mb2),
            2 * np.imag(a * b),
            np.imag(a2pb2),
            np.real(a2mb2),
            -2 * np.real(a * b),
            2 * np.imag(a * np.conj(b)),
            2 * np.real(a * np.conj(b)),
            np.real(a * np.conj(a)) - np.real(b * np.conj(b)),
        ]
    )

    scale = 1.0

    m33 = m33_linear.reshape([3, 3])
    m44 = np.zeros((4, 4))
    m44[0:3, 0:3] = m33 * scale
    m44[0:3, 3] = center_scaled.flatten()
    m44[3, 3] = 1
    transform = np.linalg.inv(m44)

    return transform
