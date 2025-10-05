# This function is highly amenable to JAX conversion.
# All NumPy operations involving complex numbers, array reshaping, and linear algebra can be directly replaced with their `jax.numpy` equivalents for potential performance gains and automatic differentiation.


import chex
import jax.numpy as jnp

from ZMPY3D_JAX.config import COMPLEX_DTYPE

# create a type alias for COMPLEX_DTYPE
Complex = COMPLEX_DTYPE


def get_transform_matrix_from_ab_list02(
    a: Complex, b: Complex, center_scaled: chex.Array
) -> chex.Array:
    """Constructs a 4x4 transformation matrix (including rotation and translation)
    from complex 'a' and 'b' coefficients and a 'center_scaled' vector.
    This matrix is used for superimposition.

    Args:
        a (COMPLEX_DTYPE): The complex 'a' coefficient.
        b (COMPLEX_DTYPE): The complex 'b' coefficient.
        center_scaled (np.ndarray): A 1D NumPy array representing the scaled center of the molecule.

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    a2pb2 = a**2 + b**2
    a2mb2 = a**2 - b**2

    m33_linear = jnp.array(
        [
            jnp.real(a2pb2),
            -jnp.imag(a2mb2),
            2 * jnp.imag(a * b),
            jnp.imag(a2pb2),
            jnp.real(a2mb2),
            -2 * jnp.real(a * b),
            2 * jnp.imag(a * jnp.conj(b)),
            2 * jnp.real(a * jnp.conj(b)),
            jnp.real(a * jnp.conj(a)) - jnp.real(b * jnp.conj(b)),
        ]
    )

    scale = 1.0

    m33 = m33_linear.reshape((3, 3))

    # build a 4x4 matrix immutably using JAX `.at` updates
    # choose a float dtype compatible with m33 / center_scaled
    dtype = m33.dtype
    m44 = jnp.zeros((4, 4), dtype=dtype)
    m44 = m44.at[0:3, 0:3].set(m33 * scale)
    m44 = m44.at[0:3, 3].set(jnp.ravel(center_scaled))
    m44 = m44.at[3, 3].set(1)

    transform = jnp.linalg.inv(m44)

    return transform
