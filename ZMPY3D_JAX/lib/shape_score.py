"""Compiled device-side reductions for descriptor and geometric shape scores."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from .descriptor_assembly import DescriptorVector


@jax.jit
def calculate_shape_scores(
    descriptor_a: DescriptorVector,
    descriptor_b: DescriptorVector,
    geo_a: chex.Array,
    geo_b: chex.Array,
    zm_indices: chex.Array,
    zm_weights: chex.Array,
    geo_weights: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Return scaled geometric and Zernike scores as JAX scalars."""
    zm_indices = jnp.asarray(zm_indices, dtype=jnp.int32).reshape(-1)
    zm_weights = jnp.asarray(zm_weights).reshape(-1)
    selected_valid = (
        descriptor_a.is_valid[zm_indices] & descriptor_b.is_valid[zm_indices]
    )
    zm_score = jnp.sum(
        jnp.abs(
            descriptor_a.values[zm_indices] - descriptor_b.values[zm_indices]
        )
        * zm_weights
    )
    zm_score = jnp.where(jnp.all(selected_valid), zm_score, jnp.nan)

    geo_a = jnp.asarray(geo_a).reshape(-1, 1)
    geo_b = jnp.asarray(geo_b).reshape(-1, 1)
    geo_score = jnp.sum(
        jnp.asarray(geo_weights).reshape(-1, 1)
        * (2 * jnp.abs(geo_a - geo_b) / (1 + jnp.abs(geo_a) + jnp.abs(geo_b)))
    )
    return (6.6 - geo_score) / 6.6 * 100.0, (9.0 - zm_score) / 9.0 * 100.0
