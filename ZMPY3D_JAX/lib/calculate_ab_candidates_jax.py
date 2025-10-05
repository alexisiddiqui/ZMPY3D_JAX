import chex
import jax
import jax.numpy as jnp

from ZMPY3D_JAX.lib.eigen_root import batched_eigen_root


def compute_ab_candidates_jax(
    z_moment_raw: chex.Array, abconj_sol: chex.Array, ind_real: int
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """
    Compute ALL candidate a/b values with validity mask.
    Returns fixed-size arrays - filtering happens outside JIT.
    """
    k_re = jnp.real(abconj_sol)
    k_im = jnp.imag(abconj_sol)
    k_im2, k_re2 = k_im**2, k_re**2
    k_im3, k_im4, k_re4 = k_im**3, k_im**4, k_re**4

    # Extract moments at this order
    f20 = jnp.real(z_moment_raw[ind_real, 2, 0])
    f21 = z_moment_raw[ind_real, 2, 1]
    f22 = z_moment_raw[ind_real, 2, 2]
    f21_im, f21_re = jnp.imag(f21), jnp.real(f21)
    f22_im, f22_re = jnp.imag(f22), jnp.real(f22)

    # Vectorized coefficient calculation
    coef4 = (
        4 * f22_re * k_im * (-1 + k_im2 - 3 * k_re2)
        - 4 * f22_im * k_re * (1 - 3 * k_im2 + k_re2)
        - 2 * f21_re * k_im * k_re * (-3 + k_im2 + k_re2)
        + 2 * f20 * k_im * (-1 + k_im2 + k_re2)
        + f21_im * (1 - 6 * k_im2 + k_im2**2 - k_re2**2)
    )
    coef3 = 2 * (
        -4 * f22_im * (k_im + k_im3 - 3 * k_im * k_re2)
        + f21_re * (-1 + k_im4 + 6 * k_re2 - k_re4)
        + 2
        * k_re
        * (
            f22_re * (2 + 6 * k_im2 - 2 * k_re2)
            + f21_im * k_im * (-3 + k_im2 + k_re2)
            + f20 * (-1 + k_im2 + k_re2)
        )
    )

    # Stack coefficients and solve in batch
    bimbre_coef = jnp.stack([coef4, coef3, jnp.zeros_like(coef4), coef3, -coef4], axis=1)
    bimbre_sol = batched_eigen_root(bimbre_coef)
    bimbre_sol_real = jnp.real(bimbre_sol)

    # Compute ALL a/b values (fixed shape)
    k_im2_exp = k_im2[:, None]
    k_re2_exp = k_re2[:, None]
    abconj_sol_exp = abconj_sol[:, None]

    bre = 1 / jnp.sqrt((1 + k_im2_exp + k_re2_exp) * (1 + bimbre_sol_real**2))
    bim = bimbre_sol_real * bre
    b = jax.lax.complex(bre, bim)
    a = abconj_sol_exp * jnp.conj(b)

    # Compute validity mask but DON'T filter yet
    is_valid = jnp.abs(bimbre_sol_real) > 1e-7

    # Return everything - filtering happens outside JIT
    return a.flatten(), b.flatten(), is_valid.flatten()
