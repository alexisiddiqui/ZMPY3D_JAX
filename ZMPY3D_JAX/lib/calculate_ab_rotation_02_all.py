# The initial `abconj_coef` calculation and `eigen_root` call are JAX-compatible.
# Inside `get_ab_list_by_ind_real`:
# All element-wise numerical operations and power calculations are JAX-compatible.
# The loop for `bimbre_sol_real` and the subsequent loop for `a_list`/`b_list` would need to be vectorized using `jax.vmap` or `jax.lax.scan`.
# `np.vectorize(complex)` can be replaced by direct JAX complex number construction.
# `np.concatenate` is available in `jax.numpy`.
# The outer loop over `ind_real_all` would also need to be vectorized or `jax.lax.scan`-ed.
# This function is numerically intensive and would greatly benefit from JAX transformation, but requires careful handling of loops and conditional logic.


from typing import List

import numpy as np

from .eigen_root import eigen_root


def calculate_ab_rotation_02_all(
    z_moment_raw: np.ndarray, target_order2_norm_rotate: int
) -> List[np.ndarray]:
    """Calculates all possible 'a' and 'b' Cayley-Klein rotation parameters across multiple orders.

    This function extends the single-order normalization (calculate_ab_rotation_02) to compute
    rotation parameters for all even orders from 2 to the maximum available in the input moments.
    This enables multi-order Canterakis Norm descriptors as described in Guzenko et al. (2020).

    Mathematical Background:
    ------------------------
    The complete Canterakis Norms (CNs) approach uses normalizations at multiple orders (n=2,3,4,5,...)
    to create a more robust rotation-invariant descriptor. Each normalization order provides different
    alignment properties:
    - Order 2: Equivalent to principal component alignment
    - Higher orders: Match progressively finer structural details

    The BioZernike descriptor averages absolute values across multiple normalization orders
    to handle symmetry ambiguities and create a versatile shape descriptor (see Fig 3d in reference).

    Algorithm Steps:
    ---------------
    1. Determine initial rotation candidates from target_order2_norm_rotate (using abconj_sol)
    2. For each even order from 2 to n_max:
        a. Extract Zernike moments at that order
        b. Compute polynomial coefficients (Equations 11-12 from reference)
        c. Solve for rotation parameters for each candidate
        d. Filter numerically stable solutions
    3. Return list of parameter arrays, one per order

    Args:
        z_moment_raw (np.ndarray): 3D array of raw Zernike moments with shape (n_max+1, l_max+1, m_max+1)
            where indices represent [n, l, m] quantum numbers
        target_order2_norm_rotate (int): Initial target order for computing rotation candidates.
            Must be >= 2 and determines the primary normalization constraint.

    Returns:
        List[np.ndarray]: List of arrays, one per ind_real order (2, 4, 6, ..., n_max).
            Each array has shape (n_solutions, 2) with complex [a, b] rotation parameters.
            Different orders may yield different numbers of valid solutions.

    References:
        Guzenko, D., Burley, S. K., & Duarte, J. M. (2020). Real time structural search of the
        Protein Data Bank. PLoS Computational Biology, 16(7), e1007970.
        https://doi.org/10.1371/journal.pcbi.1007970

        See specifically:
        - Figure 3d: Multi-order CN descriptor construction
        - Methods section: "CNs of orders n = 2, 3, 4, 5 are computed..."
        - Equations 11-12: Polynomial coefficient derivations

    Notes:
        - Only processes even ind_real orders (2, 4, 6, ...) due to Zernike moment properties
        - Each order may have different numbers of solutions due to numerical filtering
        - Solutions can be used for multi-order alignment or averaged for robust descriptors
        - The nested function get_ab_list_by_ind_real() is JAX-compatible with vectorization

    JAX Compatibility Notes:
        - Initial abconj_coef calculation and eigen_root call are JAX-compatible
        - Element-wise operations and power calculations are JAX-compatible
        - Loops over bimbre_sol_real and ind_real_all require vectorization via jax.vmap or jax.lax.scan
        - np.vectorize(complex) should be replaced with direct JAX complex construction
        - Conditional filtering (is_abs_bimre_good) requires careful handling in JAX
    """
    if target_order2_norm_rotate % 2 == 0:
        abconj_coef = [
            z_moment_raw[target_order2_norm_rotate, 2, 2],
            -z_moment_raw[target_order2_norm_rotate, 2, 1],
            z_moment_raw[target_order2_norm_rotate, 2, 0],
            np.conj(z_moment_raw[target_order2_norm_rotate, 2, 1]),
            np.conj(z_moment_raw[target_order2_norm_rotate, 2, 2]),
        ]
        abconj_sol = eigen_root(abconj_coef)
        n_abconj = 4
    else:
        abconj_coef = [
            z_moment_raw[target_order2_norm_rotate, 1, 1],
            -z_moment_raw[target_order2_norm_rotate, 1, 0],
            -np.conj(z_moment_raw[target_order2_norm_rotate, 1, 1]),
        ]
        abconj_sol = eigen_root(abconj_coef)
        n_abconj = 2

    def get_ab_list_by_ind_real(
        z_moment_raw: np.ndarray, abconj_sol: np.ndarray, ind_real: int
    ) -> np.ndarray:
        k_re = np.real(abconj_sol)
        k_im = np.imag(abconj_sol)
        k_im2 = np.imag(abconj_sol) ** 2
        k_re2 = np.real(abconj_sol) ** 2
        k_im3 = np.imag(abconj_sol) ** 3
        k_im4 = np.imag(abconj_sol) ** 4
        k_re4 = np.real(abconj_sol) ** 4

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
            bre = 1 / np.sqrt((1 + k_im2[i] + k_re2[i]) * (1 + np.power(bimbre_sol_real[i], 2)))
            bim = bimbre_sol_real[i] * bre
            b = np.vectorize(complex)(bre, bim)
            a = abconj_sol[i] * np.conj(b)
            a_list.append(a[is_abs_bimre_good[i]])
            b_list.append(b[is_abs_bimre_good[i]])

        ab_list = np.concatenate(
            (np.concatenate(a_list).reshape(-1, 1), np.concatenate(b_list).reshape(-1, 1)), axis=1
        )

        return ab_list

    ind_real_all = np.arange(2, z_moment_raw.shape[0] + 1, 2)
    ab_list_all = [None] * len(ind_real_all)

    for i, ind_real in enumerate(ind_real_all):
        ab_list_all[i] = get_ab_list_by_ind_real(z_moment_raw, abconj_sol, ind_real)

    return ab_list_all
