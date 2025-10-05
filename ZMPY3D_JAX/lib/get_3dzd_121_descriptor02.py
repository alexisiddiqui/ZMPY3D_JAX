# All numerical operations (`jnp.abs`, `**`, `jnp.sum`, `jnp.sqrt`) have direct equivalents in `jax.numpy`.
# Array slicing and conditional assignments can be handled with `jax.numpy.where` or `jax.lax.select`.
# This function is fully JAX-compatible and can be used with jit, grad, and vmap.


import chex
import jax.numpy as jnp

import ZMPY3D_JAX.config as _config


def get_3dzd_121_descriptor02(z_moment_scaled: chex.Array) -> chex.Array:
    """Calculates the 3D Zernike Descriptor (3DZD) 121 invariant from scaled Zernike moments.
    This invariant is a rotation-invariant descriptor of molecular shape.

    Args:
        z_moment_scaled (Array): An array of scaled Zernike moments.

    Returns:
        chex.Array: A JAX array representing the 3DZD 121 invariant.
    """
    z_moment_scaled = jnp.asarray(z_moment_scaled, dtype=_config.COMPLEX_DTYPE)
    z_moment_scaled = jnp.where(jnp.isnan(z_moment_scaled), 0, z_moment_scaled)
    z_moment_scaled_norm = jnp.abs(z_moment_scaled) ** 2

    z_moment_scaled_norm_positive = jnp.sum(z_moment_scaled_norm, axis=2)

    z_moment_scaled_norm = z_moment_scaled_norm.at[:, :, 0].set(0)
    z_moment_scaled_norm_negative = jnp.sum(z_moment_scaled_norm, axis=2)

    zm_3dzd_invariant = jnp.sqrt(z_moment_scaled_norm_positive + z_moment_scaled_norm_negative)
    zm_3dzd_invariant = jnp.where(zm_3dzd_invariant < 1e-20, jnp.nan, zm_3dzd_invariant)

    return zm_3dzd_invariant
