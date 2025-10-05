import chex
import jax
import jax.numpy as jnp

# Import config lazily inside the function to respect runtime configuration (e.g. tests enabling x64)
import ZMPY3D_JAX.config as _config


def eigen_root(poly_coefficient_list: chex.Array) -> chex.Array:
    """Calculates the roots of a polynomial given its coefficients by constructing
    a companion matrix and finding its eigenvalues.
    """
    COMPLEX_DTYPE = _config.COMPLEX_DTYPE
    coef = jnp.asarray(poly_coefficient_list, dtype=COMPLEX_DTYPE).reshape(-1)
    n = coef.shape[0] - 1

    if n <= 0:
        return jnp.asarray([], dtype=COMPLEX_DTYPE)

    if coef[0] == 0:
        raise ValueError("Leading coefficient must be non-zero for companion matrix construction")

    # build companion matrix with ones on subdiagonal (concise)
    m = jnp.diag(jnp.ones(n - 1, dtype=COMPLEX_DTYPE), k=-1)
    m = m.at[0, :].set(-coef[1:] / coef[0])

    return jnp.linalg.eigvals(m)


# Vectorized version for batching
batched_eigen_root = jax.vmap(eigen_root)
