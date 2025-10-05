# The initial complex arithmetic and logarithmic power calculations are highly suitable for JAX.
# The conditional logic for `f_exp` can be handled with `jax.numpy.where`.
# The loop over `zm_rotated_list` needs to be vectorized or transformed using `jax.lax.scan` or `jax.vmap`.
# The `np.add.at` operation is a major challenge. In JAX, this would typically be replaced by `jax.ops.index_add` or by re-thinking the accumulation as a functional update, possibly using `jax.scipy.special.logsumexp` if the sum is over exponentials.
# Array reshaping and transposing are directly supported by `jax.numpy`.

from typing import List

import numpy as np


def calculate_zm_by_ab_rotation01(
    z_moment_raw: np.ndarray,
    binomial_cache: np.ndarray,
    ab_list: np.ndarray,
    max_order: int,
    clm_cache: np.ndarray,
    s_id: np.ndarray,
    n: np.ndarray,
    l: np.ndarray,
    m: np.ndarray,
    mu: np.ndarray,
    k: np.ndarray,
    is_nlm_value: np.ndarray,
) -> List[np.ndarray]:
    """Calculates rotated Zernike moments based on raw Zernike moments and rotation coefficients (`a`, `b`).
    It uses pre-computed binomial and CLM (Clebsch-Gordan coefficients) caches.

    Args:
        z_moment_raw (np.ndarray): A 3D NumPy array of raw Zernike moments.
        binomial_cache (np.ndarray): A cache of binomial coefficients.
        ab_list (np.ndarray): A NumPy array where each row contains complex 'a' and 'b' rotation coefficients.
        max_order (int): The maximum order of Zernike moments.
        clm_cache (np.ndarray): A cache of Clebsch-Gordan coefficients.
        s_id (np.ndarray): A NumPy array of indices for updating Zernike moments.
        n (np.ndarray): A NumPy array of n-values for Zernike moments.
        l (np.ndarray): A NumPy array of l-values for Zernike moments.
        m (np.ndarray): A NumPy array of m-values for Zernike moments.
        mu (np.ndarray): A NumPy array of mu-values for Zernike moments.
        k (np.ndarray): A NumPy array of k-values for Zernike moments.
        is_nlm_value (np.ndarray): A boolean NumPy array indicating valid nlm values.

    Returns:
        list: A list of NumPy arrays, where each array contains the rotated Zernike moments.
    """
    zm_rotated_list = [None] * len(ab_list)

    a = ab_list[:, 0]
    b = ab_list[:, 1]
    a = a.flatten()
    b = b.flatten()

    aac = np.real(a * np.conj(a)).astype(np.complex128)
    bbc = np.real(b * np.conj(b)).astype(np.complex128)
    bbcaac = -bbc / aac

    abc = -(a / np.conj(b))
    ab = a / b

    bbcaac_pow_k_list = np.log(bbcaac)[:, None] * np.arange(max_order + 1)
    aac_pow_l_list = np.log(aac)[:, None] * np.arange(max_order + 1)
    ab_pow_m_list = np.log(ab)[:, None] * np.arange(max_order + 1)
    abc_pow_mu_list = np.log(abc)[:, None] * np.arange(-max_order, max_order + 1)

    f_exp = np.zeros(len(s_id), dtype=np.complex128)
    f_exp[mu >= 0] = z_moment_raw[n[mu >= 0], l[mu >= 0], mu[mu >= 0]]
    f_exp[(mu < 0) & (mu % 2 == 0)] = np.conj(
        z_moment_raw[
            n[(mu < 0) & (mu % 2 == 0)], l[(mu < 0) & (mu % 2 == 0)], -mu[(mu < 0) & (mu % 2 == 0)]
        ]
    )
    f_exp[(mu < 0) & (mu % 2 != 0)] = -np.conj(
        z_moment_raw[
            n[(mu < 0) & (mu % 2 != 0)], l[(mu < 0) & (mu % 2 != 0)], -mu[(mu < 0) & (mu % 2 != 0)]
        ]
    )

    f_exp = np.log(f_exp)

    max_n = max_order + 1
    clm = clm_cache[l * max_n + m].astype(np.complex128)
    clm = clm.flatten()

    bin = binomial_cache[l - mu, k - mu].astype(np.complex128) + binomial_cache[
        l + mu, k - m
    ].astype(np.complex128)

    for zm_i in range(len(zm_rotated_list)):
        al = aac_pow_l_list[zm_i, l]
        al = al.flatten()

        abpm = ab_pow_m_list[zm_i, m]
        abpm = abpm.flatten()

        amu = abc_pow_mu_list[zm_i, max_order + mu]
        amu = amu.flatten()

        bbk = bbcaac_pow_k_list[zm_i, k]
        bbk = bbk.flatten()

        nlm = f_exp + al + clm + abpm + amu + bbk + bin

        z_nlm = np.zeros(is_nlm_value.shape, dtype=np.complex128)
        np.add.at(z_nlm, s_id, np.exp(nlm))

        zm = np.full((np.prod(z_moment_raw.shape),), np.nan, dtype=np.complex128)
        zm[is_nlm_value] = z_nlm
        zm = zm.reshape(z_moment_raw.shape)
        zm = np.transpose(zm, (2, 1, 0))

        zm_rotated_list[zm_i] = zm

    return zm_rotated_list
