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

from .calculate_ab_candidates_jax import compute_ab_candidates_jax
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
        - This function now uses a JAX-based implementation for candidate computation.
        - The core logic is in `compute_ab_candidates_jax`.
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
        """Computes a/b candidates for a single order using JAX-based implementation."""
        a, b, is_valid = compute_ab_candidates_jax(z_moment_raw, abconj_sol, ind_real)

        # Filter invalid solutions and reshape
        a_flat = a[is_valid]
        b_flat = b[is_valid]

        if a_flat.size == 0:
            return np.empty((0, 2), dtype=complex)

        return np.stack([a_flat, b_flat], axis=1)

    ind_real_all = np.arange(2, z_moment_raw.shape[0], 2)
    ab_list_all = [None] * len(ind_real_all)

    for i, ind_real in enumerate(ind_real_all):
        ab_list_all[i] = get_ab_list_by_ind_real(z_moment_raw, abconj_sol, ind_real)

    return ab_list_all