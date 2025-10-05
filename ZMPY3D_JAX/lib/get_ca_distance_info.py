# All NumPy operations (`np.mean`, `np.sqrt`, `np.sum`, `**`, `np.percentile`, `np.std`) have direct equivalents in `jax.numpy` and are highly suitable for JAX transformation.
# This function would benefit significantly from JAX's numerical acceleration and automatic differentiation capabilities.

from typing import Tuple

import chex
import jax.numpy as jnp

from ZMPY3D_JAX.config import FLOAT_DTYPE


def get_ca_distance_info(xyz: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Calculates various geometric descriptors from C-alpha (CA) atom coordinates,
    including percentiles of distances to the center, standard deviation of distances,
    skewness (s), and kurtosis (k).
    """
    xyz = jnp.asarray(xyz, dtype=FLOAT_DTYPE)

    xyz_center = jnp.mean(xyz, axis=0)
    xyz_dist2center = jnp.sqrt(jnp.sum((xyz - xyz_center) ** 2, axis=1))

    percentiles_for_geom = jnp.array(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0], dtype=FLOAT_DTYPE
    )
    percentile_list = jnp.percentile(xyz_dist2center, percentiles_for_geom)
    percentile_list = percentile_list.reshape(-1, 1)

    std_xyz_dist2center = jnp.std(xyz_dist2center, ddof=1)

    n = xyz_dist2center.shape[0]
    n_f = jnp.asarray(n, dtype=FLOAT_DTYPE)
    mean_distance = jnp.mean(xyz_dist2center)

    # Avoid division by zero in case std is zero
    std_safe = jnp.where(std_xyz_dist2center == 0, 1.0, std_xyz_dist2center)

    s = (n_f / ((n_f - 1.0) * (n_f - 2.0))) * jnp.sum(
        ((xyz_dist2center - mean_distance) / std_safe) ** 3
    )

    fourth_moment = jnp.sum(((xyz_dist2center - mean_distance) / std_safe) ** 4)
    k = n_f * (n_f + 1.0) / ((n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0)) * fourth_moment - 3.0 * (
        n_f - 1.0
    ) ** 2 / ((n_f - 2.0) * (n_f - 3.0))

    return percentile_list, std_xyz_dist2center, s, k
