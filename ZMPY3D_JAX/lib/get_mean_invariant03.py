# All operations (`np.stack`, `np.abs`, `np.mean`, `np.std`) have direct equivalents in `jax.numpy` and are highly suitable for JAX transformation.
# This function would be very efficient under JAX.

from typing import Sequence, Tuple

import chex
import jax
import jax.numpy as jnp

import ZMPY3D_JAX.config as _config


@jax.jit
def _get_mean_invariant_batch(zm_batch: chex.Array) -> Tuple[chex.Array, chex.Array]:
    all_zm = jnp.abs(zm_batch)
    return jnp.mean(all_zm, axis=0), jnp.std(all_zm, axis=0, ddof=1)


def get_mean_invariant03(
    zm_list: Sequence[chex.Array] | chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Calculate rotation means/stds from a legacy list or ``(r, m, l, n)`` batch."""
    if hasattr(zm_list, "ndim") and zm_list.ndim == 4:
        batch = jnp.asarray(zm_list, dtype=_config.COMPLEX_DTYPE)
    else:
        batch = jnp.stack(
            [jnp.asarray(z, dtype=_config.COMPLEX_DTYPE) for z in zm_list], axis=0
        )
    return _get_mean_invariant_batch(batch)
