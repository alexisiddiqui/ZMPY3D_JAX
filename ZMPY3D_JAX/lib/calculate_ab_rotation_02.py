# The initial `abconj_coef` calculation and `eigen_root` call are JAX-compatible.
# All element-wise numerical operations and power calculations are JAX-compatible.
# The loop for `bimbre_sol_real` and the subsequent loop for `a_list`/`b_list` would need to be vectorized using `jax.vmap` or `jax.lax.scan`.
# `np.vectorize(complex)` can be replaced by direct JAX complex number construction.
# `np.concatenate` is available in `jax.numpy`.
# This function is numerically intensive and would greatly benefit from JAX transformation, but requires careful handling of loops and conditional logic.


import chex
import jax.numpy as jnp
import numpy as np

from ZMPY3D_JAX import config as _config

from .calculate_ab_candidates_jax import calculate_ab_rotation_candidates


def calculate_ab_rotation_02(
    z_moment_raw: chex.Array, target_order2_norm_rotate: int
) -> chex.Array:
    """Calculates 'a' and 'b' Cayley-Klein rotation parameters for 3D Zernike moment normalization.

    This function implements the complete rotation-invariant 3D Zernike moment normalization
    procedure (Canterakis Norms) as described in Guzenko et al. (2020). The method solves
    polynomial equations to find rotation parameters that orient a structure into a standard
    position, making Zernike moments rotationally invariant.

    Mathematical Background:
    ------------------------
    The Cayley-Klein parameters a and b define a rotation R(a,b) with the constraint
    ``|a|² + |b|² = 1``.
    The normalization procedure fixes rotational degrees of freedom by setting selected moments
    to predefined values (typically zero), which requires solving a system of polynomial equations.

    For even target orders: Solves 4th degree polynomial (Eq. 11) followed by 2nd degree (Eq. 12)
    For odd target orders: Solves 2nd degree polynomial with modified coefficients

    The coefficients coef4 and coef3 correspond to Equations 11-12 in the reference paper.
    The asymmetry in power terms (k_im present, k_re absent) is mathematically correct
    and arises from the specific rotation properties of spherical harmonics in 3D Zernike moments.

    Algorithm Steps:
    ---------------
    1. Extract appropriate Zernike moment coefficients based on target order parity
    2. Solve for rotation parameter candidates (abconj_sol) using eigen_root
    3. Compute polynomial coefficients for the second normalization constraint
    4. Solve for b_im/b_re ratio (bimbre) for each candidate rotation
    5. Construct full a and b parameters, filtering numerically unstable solutions

    Args:
        z_moment_raw (np.ndarray): 3D array of raw Zernike moments with shape (n_max+1, l_max+1, m_max+1)
            where indices represent [n, l, m] quantum numbers
        target_order2_norm_rotate (int): Target order for normalization (determines which moments
            are set to zero). Must be >= 2.

    Returns:
        np.ndarray: Array of shape (n_solutions, 2) where each row contains complex [a, b] parameters
            representing valid Cayley-Klein rotation parameters satisfying ``|a|² + |b|² = 1``.
            Multiple solutions may exist due to symmetry ambiguities.

    References:
        Guzenko, D., Burley, S. K., & Duarte, J. M. (2020). Real time structural search of the
        Protein Data Bank. PLoS Computational Biology, 16(7), e1007970.
        https://doi.org/10.1371/journal.pcbi.1007970

    See specifically:
        - Equation 11: 4th degree polynomial in t = a/conj(a)
        - Equation 12: 2nd degree polynomial in b_im/b_re
        - Equations 5-9: Cayley-Klein rotation formulation

    Notes:
        - The function uses ind_real=2 hardcoded for the second normalization constraint
        - Solutions with |b_im/b_re| < 1e-7 are filtered as numerically unstable
        - The method preserves all information while achieving rotation invariance
    """

    candidates = calculate_ab_rotation_candidates(
        z_moment_raw, target_order2_norm_rotate
    )
    pairs = np.asarray(candidates.pairs)
    is_valid = np.asarray(candidates.is_valid)
    compact = pairs[is_valid]
    return jnp.asarray(compact, dtype=_config.COMPLEX_DTYPE).reshape((-1, 2))
