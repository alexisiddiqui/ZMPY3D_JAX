import chex
import jax
import jax.numpy as jnp

# Import config lazily inside the function to respect runtime configuration (e.g. tests enabling x64)
import ZMPY3D_JAX.config as _config


@jax.jit
def _eigen_root_jax(coef: chex.Array) -> chex.Array:
    """Fixed-shape compiled companion-matrix eigensolve."""
    complex_dtype = jnp.result_type(coef.dtype, jnp.complex64)
    coef = jnp.asarray(coef, dtype=complex_dtype)
    n = coef.shape[0] - 1

    if n <= 0:
        return jnp.asarray([], dtype=coef.dtype)

    def true_fn(c):
        return jnp.full((n,), jnp.nan, dtype=coef.dtype)

    def false_fn(c):
        m = jnp.diag(jnp.ones(n - 1, dtype=coef.dtype), k=-1)
        m = m.at[0, :].set(-c[1:] / c[0])
        return jnp.linalg.eigvals(m)

    return jax.lax.cond(coef[0] == 0, true_fn, false_fn, coef)


def eigen_root(poly_coefficient_list: chex.Array) -> chex.Array:
    """Calculate polynomial roots through a dtype-aware compiled kernel."""
    coef = jnp.asarray(poly_coefficient_list, dtype=_config.COMPLEX_DTYPE).reshape(-1)
    return _eigen_root_jax(coef)


# Use the fixed-shape kernel directly when nested inside other compiled kernels.
batched_eigen_root = jax.vmap(_eigen_root_jax)
