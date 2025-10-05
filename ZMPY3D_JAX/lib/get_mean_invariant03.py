# All operations (`np.stack`, `np.abs`, `np.mean`, `np.std`) have direct equivalents in `jax.numpy` and are highly suitable for JAX transformation.
# This function would be very efficient under JAX.

from typing import Sequence, Tuple

import chex
import jax.numpy as jnp

from ZMPY3D_JAX.config import COMPLEX_DTYPE


def get_mean_invariant03(zm_list: Sequence[chex.Array]) -> Tuple[chex.Array, chex.Array]:
    """Calculates the mean and standard deviation of a list of Zernike moment arrays,
    typically representing different rotations of a molecule.
    """
    # ensure complex JAX arrays, stack along a new axis 3
    stacked = jnp.stack([jnp.asarray(z, dtype=COMPLEX_DTYPE) for z in zm_list], axis=3)
    all_zm = jnp.abs(stacked)

    zm_mean = jnp.mean(all_zm, axis=3)
    zm_std = jnp.std(all_zm, axis=3, ddof=1)

    return zm_mean, zm_std
