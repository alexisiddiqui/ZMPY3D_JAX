"""Fixed-shape, device-native assembly of invariant descriptor vectors."""

from typing import NamedTuple, Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np

import ZMPY3D_JAX.config as _config


class DescriptorVector(NamedTuple):
    """Structurally compact descriptor values and their data-validity mask."""

    values: chex.Array
    is_valid: chex.Array


class DescriptorAssemblyCache(NamedTuple):
    """Device-resident structural indices for a fixed maximum order."""

    max_order: int
    descriptor_indices: chex.Array
    moment_indices: chex.Array


def prepare_descriptor_assembly_cache(max_order: int) -> DescriptorAssemblyCache:
    """Prepare fixed 3DZD and established rotated-output gather indices."""
    if max_order < 0:
        raise ValueError("max_order must be non-negative")

    max_n = int(max_order) + 1
    descriptor_mask = np.zeros((max_n, max_n), dtype=bool)
    moment_mask_nlm = np.zeros((max_n, max_n, max_n), dtype=bool)
    for n in range(max_n):
        for l in range(n + 1):
            if (n - l) % 2 == 0:
                descriptor_mask[n, l] = True
                moment_mask_nlm[n, l, : l + 1] = True

    return DescriptorAssemblyCache(
        max_order=int(max_order),
        descriptor_indices=jnp.asarray(
            np.flatnonzero(descriptor_mask.reshape(-1)), dtype=jnp.int32
        ),
        moment_indices=jnp.asarray(
            np.flatnonzero(moment_mask_nlm.reshape(-1)), dtype=jnp.int32
        ),
    )


@jax.jit
def _assemble_descriptor_blocks(
    descriptor: chex.Array,
    mean_invariants: chex.Array,
    descriptor_indices: chex.Array,
    moment_indices: chex.Array,
) -> DescriptorVector:
    descriptor_values = descriptor.reshape(-1)[descriptor_indices]
    block_size = (
        mean_invariants.shape[1]
        * mean_invariants.shape[2]
        * mean_invariants.shape[3]
    )
    mean_values = mean_invariants.reshape((mean_invariants.shape[0], block_size))[
        :, moment_indices
    ]
    values = jnp.concatenate((descriptor_values, mean_values.reshape(-1)))
    return DescriptorVector(values=values, is_valid=~jnp.isnan(values))


@jax.jit
def _assemble_mean_blocks(
    mean_invariants: chex.Array, moment_indices: chex.Array
) -> DescriptorVector:
    block_size = (
        mean_invariants.shape[1]
        * mean_invariants.shape[2]
        * mean_invariants.shape[3]
    )
    values = mean_invariants.reshape((mean_invariants.shape[0], block_size))[
        :, moment_indices
    ]
    values = values.reshape(-1)
    return DescriptorVector(values=values, is_valid=~jnp.isnan(values))


def assemble_descriptor_vector(
    descriptor_3dzd: chex.Array | None,
    mean_invariants: Sequence[chex.Array] | chex.Array,
    cache: DescriptorAssemblyCache,
) -> DescriptorVector:
    """Assemble fixed structural slots without data-dependent boolean compaction.

    ``mean_invariants`` may be a sequence or a stacked ``(k, N, N, N)`` array.
    The 3DZD block, when supplied, precedes normalization blocks in the result.
    """
    if hasattr(mean_invariants, "ndim"):
        means = jnp.asarray(mean_invariants, dtype=_config.FLOAT_DTYPE)
        if means.ndim != 4:
            raise ValueError("mean_invariants must be a rank-4 array")
        max_n = means.shape[1]
    else:
        items = [jnp.asarray(item, dtype=_config.FLOAT_DTYPE) for item in mean_invariants]
        if items:
            max_n = items[0].shape[0]
            means = jnp.stack(items)
        else:
            max_n = (
                jnp.asarray(descriptor_3dzd).shape[0]
                if descriptor_3dzd is not None
                else int(cache.max_order) + 1
            )
            means = jnp.empty(
                (0, max_n, max_n, max_n), dtype=_config.FLOAT_DTYPE
            )

    if means.shape[1:] != (max_n, max_n, max_n):
        raise ValueError(
            f"mean_invariants must have shape (k, {max_n}, {max_n}, {max_n})"
        )

    if descriptor_3dzd is None:
        return _assemble_mean_blocks(means, cache.moment_indices)

    descriptor = jnp.asarray(descriptor_3dzd, dtype=_config.FLOAT_DTYPE)
    if descriptor.shape != (max_n, max_n):
        raise ValueError(f"descriptor_3dzd must have shape ({max_n}, {max_n})")
    return _assemble_descriptor_blocks(
        descriptor, means, cache.descriptor_indices, cache.moment_indices
    )


def stack_descriptor_vectors(vectors: Sequence[DescriptorVector], width: int) -> DescriptorVector:
    """Stack descriptor vectors, including a well-defined empty batch."""
    if vectors:
        return DescriptorVector(
            values=jnp.stack([item.values for item in vectors]),
            is_valid=jnp.stack([item.is_valid for item in vectors]),
        )
    return DescriptorVector(
        values=jnp.empty((0, width), dtype=_config.FLOAT_DTYPE),
        is_valid=jnp.empty((0, width), dtype=bool),
    )


def compact_descriptor_for_host(descriptor: DescriptorVector) -> np.ndarray:
    """Compact one vector at an explicit host/I/O boundary."""
    values = np.asarray(descriptor.values)
    mask = np.asarray(descriptor.is_valid)
    return values[mask]
