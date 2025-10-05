# The initial `abconj_coef` calculation and `eigen_root` call are JAX-compatible.
# All element-wise numerical operations and power calculations are JAX-compatible.
# The loop for `bimbre_sol_real` and the subsequent loop for `a_list`/`b_list` would need to be vectorized using `jax.vmap` or `jax.lax.scan`.
# `np.vectorize(complex)` can be replaced by direct JAX complex number construction.
# `np.concatenate` is available in `jax.numpy`.
# This function is numerically intensive and would greatly benefit from JAX transformation, but requires careful handling of loops and conditional logic.


import numpy as np

from .calculate_ab_candidates_jax import compute_ab_candidates_jax
from .eigen_root import eigen_root


def calculate_ab_rotation_02(
    z_moment_raw: np.ndarray, target_order2_norm_rotate: int
) -> np.ndarray:
    """Calculates 'a' and 'b' Cayley-Klein rotation parameters for 3D Zernike moment normalization.

    This function implements the complete rotation-invariant 3D Zernike moment normalization
    procedure (Canterakis Norms) as described in Guzenko et al. (2020). The method solves
    polynomial equations to find rotation parameters that orient a structure into a standard
    position, making Zernike moments rotationally invariant.

    Mathematical Background:
    ------------------------
    The Cayley-Klein parameters a and b define a rotation R(a,b) with the constraint |a| + |b| = 1.
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
            representing valid Cayley-Klein rotation parameters satisfying |a| + |b| = 1.
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
    ind_real = 2

    if target_order2_norm_rotate % 2 == 0:
        abconj_coef = [
            z_moment_raw[target_order2_norm_rotate, 2, 2],
            -z_moment_raw[target_order2_norm_rotate, 2, 1],
            z_moment_raw[target_order2_norm_rotate, 2, 0],
            np.conj(z_moment_raw[target_order2_norm_rotate, 2, 1]),
            np.conj(z_moment_raw[target_order2_norm_rotate, 2, 2]),
        ]
        abconj_sol = eigen_root(abconj_coef)
        a, b, is_valid = compute_ab_candidates_jax(z_moment_raw, abconj_sol, ind_real)

        # Filter invalid solutions and reshape
        a_flat = a[is_valid]
        b_flat = b[is_valid]

        if a_flat.size == 0:
            return np.empty((0, 2), dtype=complex)

        return np.stack([a_flat, b_flat], axis=1)
    else:
        abconj_coef = [
            z_moment_raw[target_order2_norm_rotate, 1, 1],
            -z_moment_raw[target_order2_norm_rotate, 1, 0],
            -np.conj(z_moment_raw[target_order2_norm_rotate, 1, 1]),
        ]
        abconj_sol = eigen_root(abconj_coef)

        k_re = np.real(abconj_sol)
        k_im = np.imag(abconj_sol)
        k_im2 = k_im**2
        k_re2 = k_re**2
        k_im3 = k_im**3
        k_im4 = k_im**4
        k_re4 = k_re**4

        f20 = np.real(z_moment_raw[ind_real, 2, 0])
        f21 = z_moment_raw[ind_real, 2, 1]
        f22 = z_moment_raw[ind_real, 2, 2]

        f21_im = np.imag(f21)
        f21_re = np.real(f21)
        f22_im = np.imag(f22)
        f22_re = np.real(f22)

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

        bimbre_coef = np.array([coef4, coef3, np.zeros_like(coef4), coef3, -coef4]).T

        bimbre_sol_real = [np.real(eigen_root(bc)) for bc in bimbre_coef]

        is_abs_bimre_good = [np.abs(x) > 1e-7 for x in bimbre_sol_real]

        a_list = []
        b_list = []

        for i in range(len(bimbre_sol_real)):
            bre = 1 / np.sqrt(
                (1 + k_im2[i] + k_re2[i]) * (1 + np.power(bimbre_sol_real[i], 2))
            )
            bim = bimbre_sol_real[i] * bre
            b = np.vectorize(complex)(bre, bim)
            a = abconj_sol[i] * np.conj(b)

            a_list.append(a[is_abs_bimre_good[i]])
            b_list.append(b[is_abs_bimre_good[i]])

        ab_list = np.concatenate(
            (np.concatenate(a_list).reshape(-1, 1), np.concatenate(b_list).reshape(-1, 1)),
            axis=1,
        )

        return ab_list