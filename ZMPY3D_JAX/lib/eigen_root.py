# All NumPy operations (`np.reshape`, `np.diag`, array assignment, `np.linalg.eigvals`) have direct equivalents in `jax.numpy` and `jax.numpy.linalg`.
# This function is highly suitable for JAX transformation and would benefit from JAX's optimized linear algebra routines.


import chex
import jax.numpy as jnp

# Import config lazily inside the function to respect runtime configuration (e.g. tests enabling x64)
import ZMPY3D_JAX.config as _config


def eigen_root(poly_coefficient_list: chex.Array) -> chex.Array:
    """Calculates the roots of a polynomial given its coefficients by constructing
    a companion matrix and finding its eigenvalues.
    """
    COMPLEX_DTYPE = _config.COMPLEX_DTYPE
    poly_coefficient_list = jnp.asarray(poly_coefficient_list, dtype=COMPLEX_DTYPE).reshape(-1)
    n = poly_coefficient_list.shape[0] - 1

    if n <= 0:
        return jnp.asarray([], dtype=COMPLEX_DTYPE)

    # build companion matrix with ones on subdiagonal
    m = jnp.zeros((n, n), dtype=COMPLEX_DTYPE)
    ones = jnp.ones(n - 1, dtype=COMPLEX_DTYPE)
    m = m.at[jnp.arange(1, n), jnp.arange(0, n - 1)].set(ones)

    # first row: -coeff[1:]/coeff[0]
    first_row = -poly_coefficient_list[1:] / poly_coefficient_list[0]
    m = m.at[0, :].set(first_row)

    result = jnp.linalg.eigvals(m)

    return result
