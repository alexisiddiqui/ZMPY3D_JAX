# The initial `abconj_coef` calculation and `eigen_root` call are JAX-compatible.
# Inside `get_ab_list_by_ind_real`:
# All element-wise numerical operations and power calculations are JAX-compatible.
# The loop for `bimbre_sol_real` and the subsequent loop for `a_list`/`b_list` would need to be vectorized using `jax.vmap` or `jax.lax.scan`.
# `np.vectorize(complex)` can be replaced by direct JAX complex number construction.
# `np.concatenate` is available in `jax.numpy`.
# The outer loop over `ind_real_all` would also need to be vectorized or `jax.lax.scan`-ed.
# This function is numerically intensive and would greatly benefit from JAX transformation, but requires careful handling of loops and conditional logic.


import chex
import jax
import jax.numpy as jnp
import numpy as np

from ZMPY3D_JAX import config as _config

from .calculate_ab_candidates_jax import compute_ab_candidates_jax
from .eigen_root import eigen_root


def calculate_ab_rotation_02_all_jax(
    z_moment_raw: jnp.ndarray, abconj_sol: jnp.ndarray
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """
    Fully vectorized JAX implementation - processes ALL ind_real in parallel.

    Returns:
        a_all: shape (n_orders, n_abconj, n_roots)
        b_all: shape (n_orders, n_abconj, n_roots)
        is_valid_all: shape (n_orders, n_abconj, n_roots) - boolean mask
    """
    # Compute all ind_real values (static shape)
    max_order = z_moment_raw.shape[0]
    ind_real_all = jnp.arange(2, max_order, 2)

    # Vmap over ind_real dimension
    def compute_for_order(ind_real):
        return compute_ab_candidates_jax(z_moment_raw, abconj_sol, ind_real)

    a_all, b_all, is_valid_all = jax.vmap(compute_for_order)(ind_real_all)

    return a_all, b_all, is_valid_all


def extract_valid_solutions(
    a_all: np.ndarray, b_all: np.ndarray, is_valid_all: np.ndarray
) -> list[np.ndarray]:
    """Filter and format results OUTSIDE JIT (Python loop is fine here)."""
    results = []

    for i in range(len(a_all)):
        # Flatten the (n_abconj, n_roots) dimensions
        a_flat = a_all[i].ravel()
        b_flat = b_all[i].ravel()
        valid_flat = is_valid_all[i].ravel()

        a_valid = a_flat[valid_flat]
        b_valid = b_flat[valid_flat]

        if a_valid.size == 0:
            results.append(np.empty((0, 2), dtype=complex))
        else:
            results.append(np.stack([a_valid, b_valid], axis=1))

    return results


# Usage:
def calculate_ab_rotation_02_all(
    z_moment_raw: chex.Array, target_order2_norm_rotate: int
) -> list[np.ndarray]:
    """Public API maintaining compatibility with original function."""
    z_moment_raw = jnp.asarray(z_moment_raw, dtype=_config.COMPLEX_DTYPE)

    # Step 1: Compute abconj_sol (once, on CPU is fine)
    if target_order2_norm_rotate % 2 == 0:
        abconj_coef = jnp.array(
            [
                z_moment_raw[target_order2_norm_rotate, 2, 2],
                -z_moment_raw[target_order2_norm_rotate, 2, 1],
                z_moment_raw[target_order2_norm_rotate, 2, 0],
                jnp.conj(z_moment_raw[target_order2_norm_rotate, 2, 1]),
                jnp.conj(z_moment_raw[target_order2_norm_rotate, 2, 2]),
            ],
            dtype=_config.COMPLEX_DTYPE,
        )
    else:
        abconj_coef = jnp.array(
            [
                z_moment_raw[target_order2_norm_rotate, 1, 1],
                -z_moment_raw[target_order2_norm_rotate, 1, 0],
                -jnp.conj(z_moment_raw[target_order2_norm_rotate, 1, 1]),
            ],
            dtype=_config.COMPLEX_DTYPE,
        )

    abconj_sol = eigen_root(abconj_coef)
    # Step 2: JIT-compiled parallel processing of ALL orders
    a_all, b_all, is_valid_all = calculate_ab_rotation_02_all_jax(z_moment_raw, abconj_sol)

    # Step 3: Filter outside JIT (dynamic shapes are fine here)
    return extract_valid_solutions(np.array(a_all), np.array(b_all), np.array(is_valid_all))
