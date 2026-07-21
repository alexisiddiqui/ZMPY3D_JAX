"""Deterministic reductions for sorted, contiguous segment identifiers."""

import chex
import jax
import jax.numpy as jnp


def segmented_sum_associative(
    values: chex.Array, segment_ids: chex.Array, output_size: int
) -> chex.Array:
    """Sum sorted segments without unordered scatter-add atomics."""
    starts = jnp.concatenate(
        (jnp.ones((1,), dtype=bool), segment_ids[1:] != segment_ids[:-1])
    )

    def combine(left, right):
        left_value, left_starts = left
        right_value, right_starts = right
        value = jnp.where(right_starts, right_value, left_value + right_value)
        return value, left_starts | right_starts

    prefixes, _ = jax.lax.associative_scan(combine, (values, starts))
    output_ids = jnp.arange(output_size, dtype=segment_ids.dtype)
    positions = jnp.searchsorted(segment_ids, output_ids, side="right") - 1
    safe_positions = jnp.maximum(positions, 0)
    is_present = (positions >= 0) & (segment_ids[safe_positions] == output_ids)
    return jnp.where(is_present, prefixes[safe_positions], jnp.zeros((), values.dtype))
