import chex
import jax
import jax.numpy as jnp

# Import config lazily inside the function to respect runtime configuration (e.g. tests enabling x64)
import ZMPY3D_JAX.config as _config


@jax.jit
def batched_eigen_root(coef: chex.Array) -> chex.Array:
    """Solve fixed-degree polynomials over arbitrary leading dimensions.

    One companion tensor and one batched ``eigvals`` call are used for the
    complete input.  A zero leading coefficient retains the historical
    fixed-shape all-NaN result instead of silently reducing the degree.
    """
    complex_dtype = jnp.result_type(coef.dtype, jnp.complex64)
    coef = jnp.asarray(coef, dtype=complex_dtype)
    n = coef.shape[-1] - 1

    if n <= 0:
        return jnp.empty(coef.shape[:-1] + (0,), dtype=coef.dtype)

    invalid = coef[..., 0] == 0
    safe_leading = jnp.where(invalid, jnp.ones_like(coef[..., 0]), coef[..., 0])
    base = jnp.diag(jnp.ones(n - 1, dtype=coef.dtype), k=-1)
    matrices = jnp.broadcast_to(base, coef.shape[:-1] + (n, n))
    matrices = matrices.at[..., 0, :].set(-coef[..., 1:] / safe_leading[..., None])
    roots = jnp.linalg.eigvals(matrices)
    return jnp.where(invalid[..., None], jnp.full_like(roots, jnp.nan), roots)


@jax.jit
def _eigen_root_jax(coef: chex.Array) -> chex.Array:
    """Single-polynomial compatibility wrapper around the batched solver."""
    return batched_eigen_root(coef)


def eigen_root(poly_coefficient_list: chex.Array) -> chex.Array:
    """Calculate polynomial roots through a dtype-aware compiled kernel."""
    coef = jnp.asarray(poly_coefficient_list, dtype=_config.COMPLEX_DTYPE).reshape(-1)
    return _eigen_root_jax(coef)
