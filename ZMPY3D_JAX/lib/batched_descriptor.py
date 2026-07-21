"""Fixed-shape device batching for complete Zernike descriptor calculation."""

from functools import partial
from typing import Any, Callable, Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np

import ZMPY3D_JAX.config as _config

from .calculate_ab_candidates_jax import (
    calculate_ab_rotation_candidates,
    calculate_ab_rotation_compact_candidates,
)
from .calculate_bbox_moment06 import _calculate_bbox_moment_jax
from .calculate_bbox_moment_2_zm05 import BBoxToZMCache, _calculate_bbox_moment_2_zm_jax
from .calculate_molecular_radius03 import _radius_statistics_impl
from .calculate_zm_by_ab_rotation01 import (
    ZMRotationCache,
    _calculate_zm_by_ab_rotation_jax,
)
from .descriptor_assembly import DescriptorAssemblyCache, DescriptorVector
from .get_3dzd_121_descriptor02 import _get_3dzd_121_descriptor_jax


StageExecutor = Callable[[str, Callable[[], Any]], Any]


def _execute_directly(_name: str, function: Callable[[], Any]) -> Any:
    return function()


def pad_voxel_batch(voxels: Sequence[np.ndarray]) -> np.ndarray:
    """High-side zero-pad native host voxels and stack them into one dense batch."""
    if not voxels:
        raise ValueError("voxels must be non-empty")
    normalized = [np.asarray(voxel, dtype=_config.FLOAT_DTYPE) for voxel in voxels]
    if any(voxel.ndim != 3 for voxel in normalized):
        raise ValueError("each voxel must be a rank-3 array")
    if any(0 in voxel.shape or not np.any(voxel > 0) for voxel in normalized):
        raise ValueError(
            "zero-size array to reduction operation maximum which has no identity"
        )

    padded_shape = tuple(
        max(voxel.shape[axis] for voxel in normalized) for axis in range(3)
    )
    batch = np.zeros((len(normalized), *padded_shape), dtype=_config.FLOAT_DTYPE)
    for index, voxel in enumerate(normalized):
        batch[index, : voxel.shape[0], : voxel.shape[1], : voxel.shape[2]] = voxel
    return batch


@partial(jax.jit, static_argnums=(1,))
def _calculate_bbox_moment_batch(
    voxels: chex.Array,
    max_order: int,
    x_samples: chex.Array,
    y_samples: chex.Array,
    z_samples: chex.Array,
):
    return jax.vmap(
        lambda voxel, x_sample, y_sample, z_sample: _calculate_bbox_moment_jax(
            voxel, max_order, x_sample, y_sample, z_sample
        )
    )(voxels, x_samples, y_samples, z_samples)


def _bbox_edges(voxels: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
    batch_size = voxels.shape[0]
    return tuple(
        jnp.broadcast_to(
            jnp.arange(voxels.shape[axis] + 1, dtype=_config.FLOAT_DTYPE),
            (batch_size, voxels.shape[axis] + 1),
        )
        for axis in range(1, 4)
    )


@jax.jit
def _calculate_bbox_order1_batch(voxels: chex.Array):
    return _calculate_bbox_moment_batch(voxels, 1, *_bbox_edges(voxels))


@jax.jit
def _calculate_radius_and_samples_batch(
    voxels: chex.Array,
    centers: chex.Array,
    masses: chex.Array,
    default_radius_multiplier: float,
):
    has_weight, average_radius, max_radius = jax.vmap(
        _radius_statistics_impl, in_axes=(0, 0, 0, None)
    )(voxels, centers, masses, default_radius_multiplier)
    x_samples = (
        jnp.arange(voxels.shape[1] + 1, dtype=_config.FLOAT_DTYPE)[None, :]
        - centers[:, 0, None]
    ) / average_radius[:, None]
    y_samples = (
        jnp.arange(voxels.shape[2] + 1, dtype=_config.FLOAT_DTYPE)[None, :]
        - centers[:, 1, None]
    ) / average_radius[:, None]
    z_samples = (
        jnp.arange(voxels.shape[3] + 1, dtype=_config.FLOAT_DTYPE)[None, :]
        - centers[:, 2, None]
    ) / average_radius[:, None]
    return has_weight, average_radius, max_radius, x_samples, y_samples, z_samples


@partial(jax.jit, static_argnums=(1,))
def _calculate_bbox_max_order_batch(
    voxels: chex.Array,
    max_order: int,
    x_samples: chex.Array,
    y_samples: chex.Array,
    z_samples: chex.Array,
):
    return _calculate_bbox_moment_batch(
        voxels, max_order, x_samples, y_samples, z_samples
    )


@partial(jax.jit, static_argnums=(1,))
def _calculate_bbox_to_zm_batch(
    bbox_moments: chex.Array,
    max_order: int,
    g_coefficients: chex.Array,
    pqr_indices: chex.Array,
    output_indices: chex.Array,
    clm: chex.Array,
):
    complex_moments = jnp.asarray(bbox_moments, dtype=_config.COMPLEX_DTYPE)
    return jax.vmap(
        lambda bbox_moment: _calculate_bbox_moment_2_zm_jax(
            max_order,
            g_coefficients,
            pqr_indices,
            output_indices,
            clm,
            bbox_moment,
        )
    )(complex_moments)


@partial(jax.jit, static_argnums=(1,))
def _calculate_zm_batch(
    voxels: chex.Array,
    max_order: int,
    default_radius_multiplier: float,
    g_coefficients: chex.Array,
    pqr_indices: chex.Array,
    output_indices: chex.Array,
    clm: chex.Array,
):
    masses, centers, _ = _calculate_bbox_order1_batch(voxels)
    (
        has_weight,
        _average_radius,
        _max_radius,
        x_samples,
        y_samples,
        z_samples,
    ) = _calculate_radius_and_samples_batch(
        voxels, centers, masses, default_radius_multiplier
    )
    _, _, bbox_moments = _calculate_bbox_max_order_batch(
        voxels, max_order, x_samples, y_samples, z_samples
    )
    scaled, raw = _calculate_bbox_to_zm_batch(
        bbox_moments,
        max_order,
        g_coefficients,
        pqr_indices,
        output_indices,
        clm,
    )
    return has_weight, scaled, raw


@jax.jit
def _calculate_3dzd_batch(scaled_moments: chex.Array) -> chex.Array:
    return jax.vmap(_get_3dzd_121_descriptor_jax)(scaled_moments)


@partial(jax.jit, static_argnums=(1,))
def _calculate_ab_candidates_batch(raw_moments: chex.Array, target_order: int):
    return jax.vmap(lambda raw: calculate_ab_rotation_candidates(raw, target_order))(
        raw_moments
    )


@partial(jax.jit, static_argnums=(1,))
def _calculate_ab_compact_candidates_batch(
    raw_moments: chex.Array, target_order: int
):
    return jax.vmap(
        lambda raw: calculate_ab_rotation_compact_candidates(raw, target_order)
    )(raw_moments)


@partial(jax.jit, static_argnums=(2,))
def _calculate_rotation_batch(
    raw_moments: chex.Array,
    pairs: chex.Array,
    max_order: int,
    binomial: chex.Array,
    clm: chex.Array,
    s_id: chex.Array,
    n: chex.Array,
    ell: chex.Array,
    m: chex.Array,
    mu: chex.Array,
    k: chex.Array,
    is_nlm_value: chex.Array,
) -> chex.Array:
    return jax.vmap(
        lambda raw, item_pairs: _calculate_zm_by_ab_rotation_jax(
            raw,
            binomial,
            item_pairs,
            max_order,
            clm,
            s_id,
            n,
            ell,
            m,
            mu,
            k,
            is_nlm_value,
        )
    )(raw_moments, pairs)


@jax.jit
def _calculate_masked_mean_batch(
    rotated: chex.Array, is_valid: chex.Array
) -> chex.Array:
    keep = is_valid[..., None, None, None]
    values = jnp.where(keep, jnp.abs(rotated), 0)
    valid_count = jnp.sum(is_valid, axis=1)[:, None, None, None]
    mean = jnp.sum(values, axis=1) / jnp.maximum(valid_count, 1)
    return jnp.where(valid_count > 0, mean, jnp.nan)


@partial(jax.jit, static_argnums=(1, 3))
def _calculate_normalized_mean_batch(
    raw_moments: chex.Array,
    target_order: int,
    binomial: chex.Array,
    max_order: int,
    clm: chex.Array,
    s_id: chex.Array,
    n: chex.Array,
    ell: chex.Array,
    m: chex.Array,
    mu: chex.Array,
    k: chex.Array,
    is_nlm_value: chex.Array,
) -> chex.Array:
    candidates = _calculate_ab_candidates_batch(raw_moments, target_order)
    rotated = _calculate_rotation_batch(
        raw_moments,
        candidates.pairs,
        max_order,
        binomial,
        clm,
        s_id,
        n,
        ell,
        m,
        mu,
        k,
        is_nlm_value,
    )
    return _calculate_masked_mean_batch(rotated, candidates.is_valid)


@partial(jax.jit, static_argnums=(1, 3))
def _calculate_normalized_mean_compact_batch(
    raw_moments: chex.Array,
    target_order: int,
    binomial: chex.Array,
    max_order: int,
    clm: chex.Array,
    s_id: chex.Array,
    n: chex.Array,
    ell: chex.Array,
    m: chex.Array,
    mu: chex.Array,
    k: chex.Array,
    is_nlm_value: chex.Array,
) -> chex.Array:
    candidates = _calculate_ab_compact_candidates_batch(raw_moments, target_order)
    rotated = _calculate_rotation_batch(
        raw_moments,
        candidates.pairs,
        max_order,
        binomial,
        clm,
        s_id,
        n,
        ell,
        m,
        mu,
        k,
        is_nlm_value,
    )
    return _calculate_masked_mean_batch(rotated, candidates.is_valid)


@partial(jax.jit, static_argnums=(1, 3))
def _calculate_normalized_means_parity_batch(
    raw_moments: chex.Array,
    target_orders: tuple[int, ...],
    binomial: chex.Array,
    max_order: int,
    clm: chex.Array,
    s_id: chex.Array,
    n: chex.Array,
    ell: chex.Array,
    m: chex.Array,
    mu: chex.Array,
    k: chex.Array,
    is_nlm_value: chex.Array,
) -> chex.Array:
    """Fuse same-capacity compact orders into one candidate/rotation executable."""
    pairs = []
    masks = []
    for target_order in target_orders:
        candidates = jax.vmap(
            lambda raw, order=target_order: (
                calculate_ab_rotation_compact_candidates(raw, order)
            )
        )(raw_moments)
        pairs.append(candidates.pairs)
        masks.append(candidates.is_valid)
    pair_groups = jnp.stack(pairs, axis=1)
    mask_groups = jnp.stack(masks, axis=1)

    def rotate_item(raw, item_pair_groups):
        return jax.vmap(
            lambda item_pairs: _calculate_zm_by_ab_rotation_jax(
                raw,
                binomial,
                item_pairs,
                max_order,
                clm,
                s_id,
                n,
                ell,
                m,
                mu,
                k,
                is_nlm_value,
            )
        )(item_pair_groups)

    rotated = jax.vmap(rotate_item)(raw_moments, pair_groups)
    keep = mask_groups[..., None, None, None]
    values = jnp.where(keep, jnp.abs(rotated), 0)
    valid_count = jnp.sum(mask_groups, axis=2)[..., None, None, None]
    means = jnp.sum(values, axis=2) / jnp.maximum(valid_count, 1)
    return jnp.where(valid_count > 0, means, jnp.nan)


def _calculate_normalization_means(
    raw: chex.Array,
    target_orders: tuple[int, ...],
    representation: str,
    rotation_cache: ZMRotationCache,
) -> chex.Array:
    arguments = (
        rotation_cache.binomial,
        rotation_cache.max_order,
        rotation_cache.clm,
        rotation_cache.s_id,
        rotation_cache.n,
        rotation_cache.l,
        rotation_cache.m,
        rotation_cache.mu,
        rotation_cache.k,
        rotation_cache.is_nlm_value,
    )
    if representation in ("full_fixed", "analytic_compact"):
        function = (
            _calculate_normalized_mean_batch
            if representation == "full_fixed"
            else _calculate_normalized_mean_compact_batch
        )
        return jnp.stack(
            [function(raw, target_order, *arguments) for target_order in target_orders],
            axis=1,
        )

    by_order = {}
    for parity in (0, 1):
        parity_orders = tuple(order for order in target_orders if order % 2 == parity)
        if parity_orders:
            parity_means = _calculate_normalized_means_parity_batch(
                raw, parity_orders, *arguments
            )
            for index, order in enumerate(parity_orders):
                by_order[order] = parity_means[:, index]
    return jnp.stack([by_order[order] for order in target_orders], axis=1)


@jax.jit
def _assemble_mean_batch(
    means: chex.Array, moment_indices: chex.Array
) -> DescriptorVector:
    batch_size, order_count = means.shape[:2]
    block_size = means.shape[2] * means.shape[3] * means.shape[4]
    values = means.reshape((batch_size, order_count, block_size))[:, :, moment_indices]
    values = values.reshape((batch_size, -1))
    return DescriptorVector(values=values, is_valid=~jnp.isnan(values))


@jax.jit
def _assemble_descriptor_batch(
    descriptors: chex.Array,
    means: chex.Array,
    descriptor_indices: chex.Array,
    moment_indices: chex.Array,
) -> DescriptorVector:
    descriptor_values = descriptors.reshape((descriptors.shape[0], -1))[
        :, descriptor_indices
    ]
    block_size = means.shape[2] * means.shape[3] * means.shape[4]
    mean_values = means.reshape((means.shape[0], means.shape[1], block_size))[
        :, :, moment_indices
    ].reshape((means.shape[0], -1))
    values = jnp.concatenate((descriptor_values, mean_values), axis=1)
    return DescriptorVector(values=values, is_valid=~jnp.isnan(values))


def calculate_descriptor_batch_from_voxels(
    voxels: chex.Array,
    *,
    max_order: int,
    max_target_order: int,
    mode: int,
    default_radius_multiplier: float,
    bbox_to_zm_cache: BBoxToZMCache,
    rotation_cache: ZMRotationCache,
    descriptor_cache: DescriptorAssemblyCache,
    normalization_representation: str = "analytic_compact",
) -> DescriptorVector:
    """Calculate complete descriptors for one padded, device-resident voxel batch."""
    if mode not in (0, 1, 2):
        raise ValueError("Mode must be 0, 1, or 2")
    if max_target_order < 2 or max_target_order > max_order:
        raise ValueError("max_target_order must be between 2 and max_order")
    if bbox_to_zm_cache.max_order != max_order or rotation_cache.max_order != max_order:
        raise ValueError("prepared cache maximum order does not match max_order")
    if descriptor_cache.max_order != max_order:
        raise ValueError("descriptor cache maximum order does not match max_order")
    if normalization_representation not in (
        "full_fixed",
        "analytic_compact",
        "analytic_compact_parity",
    ):
        raise ValueError("unknown normalization representation")

    voxel_batch = jnp.asarray(voxels, dtype=_config.FLOAT_DTYPE)
    if voxel_batch.ndim != 4 or voxel_batch.shape[0] == 0:
        raise ValueError(
            "voxels must have shape (batch, x, y, z) with a non-empty batch"
        )

    _, scaled, raw = _calculate_zm_batch(
        voxel_batch,
        max_order,
        default_radius_multiplier,
        bbox_to_zm_cache.g_coefficients,
        bbox_to_zm_cache.pqr_indices,
        bbox_to_zm_cache.output_indices,
        bbox_to_zm_cache.clm,
    )
    descriptors = _calculate_3dzd_batch(scaled) if mode in (1, 2) else None

    if mode in (0, 2):
        means = _calculate_normalization_means(
            raw,
            tuple(range(2, max_target_order + 1)),
            normalization_representation,
            rotation_cache,
        )
    else:
        max_n = max_order + 1
        means = jnp.empty(
            (voxel_batch.shape[0], 0, max_n, max_n, max_n),
            dtype=_config.FLOAT_DTYPE,
        )

    if descriptors is None:
        return _assemble_mean_batch(means, descriptor_cache.moment_indices)
    return _assemble_descriptor_batch(
        descriptors,
        means,
        descriptor_cache.descriptor_indices,
        descriptor_cache.moment_indices,
    )


def calculate_descriptor_batch_staged(
    voxels: chex.Array,
    *,
    max_order: int,
    max_target_order: int,
    mode: int,
    default_radius_multiplier: float,
    bbox_to_zm_cache: BBoxToZMCache,
    rotation_cache: ZMRotationCache,
    descriptor_cache: DescriptorAssemblyCache,
    stage_executor: StageExecutor | None = None,
) -> tuple[DescriptorVector, dict[int, Any]]:
    """Run the batch pipeline through independently synchronizable device stages."""
    if mode not in (0, 1, 2):
        raise ValueError("Mode must be 0, 1, or 2")
    if max_target_order < 2 or max_target_order > max_order:
        raise ValueError("max_target_order must be between 2 and max_order")
    if bbox_to_zm_cache.max_order != max_order or rotation_cache.max_order != max_order:
        raise ValueError("prepared cache maximum order does not match max_order")
    if descriptor_cache.max_order != max_order:
        raise ValueError("descriptor cache maximum order does not match max_order")

    voxel_batch = jnp.asarray(voxels, dtype=_config.FLOAT_DTYPE)
    if voxel_batch.ndim != 4 or voxel_batch.shape[0] == 0:
        raise ValueError(
            "voxels must have shape (batch, x, y, z) with a non-empty batch"
        )
    execute = _execute_directly if stage_executor is None else stage_executor

    masses, centers, _ = execute(
        "bbox_order1", lambda: _calculate_bbox_order1_batch(voxel_batch)
    )
    (
        _has_weight,
        _average_radius,
        _max_radius,
        x_samples,
        y_samples,
        z_samples,
    ) = execute(
        "radius_and_samples",
        lambda: _calculate_radius_and_samples_batch(
            voxel_batch, centers, masses, default_radius_multiplier
        ),
    )
    _, _, bbox_moments = execute(
        "bbox_max_order",
        lambda: _calculate_bbox_max_order_batch(
            voxel_batch, max_order, x_samples, y_samples, z_samples
        ),
    )
    scaled, raw = execute(
        "bbox_to_zm",
        lambda: _calculate_bbox_to_zm_batch(
            bbox_moments,
            max_order,
            bbox_to_zm_cache.g_coefficients,
            bbox_to_zm_cache.pqr_indices,
            bbox_to_zm_cache.output_indices,
            bbox_to_zm_cache.clm,
        ),
    )

    descriptors = (
        execute("descriptor_3dzd", lambda: _calculate_3dzd_batch(scaled))
        if mode in (1, 2)
        else None
    )
    candidates_by_order: dict[int, Any] = {}
    means_by_order = []
    if mode in (0, 2):
        for target_order in range(2, max_target_order + 1):
            candidates = execute(
                f"ab_candidates_order_{target_order}",
                lambda target_order=target_order: _calculate_ab_candidates_batch(
                    raw, target_order
                ),
            )
            candidates_by_order[target_order] = candidates
            rotated = execute(
                f"zm_rotation_order_{target_order}",
                lambda candidates=candidates: _calculate_rotation_batch(
                    raw,
                    candidates.pairs,
                    max_order,
                    rotation_cache.binomial,
                    rotation_cache.clm,
                    rotation_cache.s_id,
                    rotation_cache.n,
                    rotation_cache.l,
                    rotation_cache.m,
                    rotation_cache.mu,
                    rotation_cache.k,
                    rotation_cache.is_nlm_value,
                ),
            )
            means_by_order.append(
                execute(
                    f"mean_invariant_order_{target_order}",
                    lambda rotated=rotated, candidates=candidates: (
                        _calculate_masked_mean_batch(rotated, candidates.is_valid)
                    ),
                )
            )

    def assemble() -> DescriptorVector:
        max_n = max_order + 1
        means = (
            jnp.stack(means_by_order, axis=1)
            if means_by_order
            else jnp.empty(
                (voxel_batch.shape[0], 0, max_n, max_n, max_n),
                dtype=_config.FLOAT_DTYPE,
            )
        )
        if descriptors is None:
            return _assemble_mean_batch(means, descriptor_cache.moment_indices)
        return _assemble_descriptor_batch(
            descriptors,
            means,
            descriptor_cache.descriptor_indices,
            descriptor_cache.moment_indices,
        )

    result = execute("descriptor_assembly", assemble)
    return result, candidates_by_order
