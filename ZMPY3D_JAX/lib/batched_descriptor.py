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
    calculate_ab_rotation_compact_candidate_group,
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
from .direct_moment_backend import (
    DirectMomentCache,
    calculate_direct_moments,
    pack_occupied_voxels,
)
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
    float_dtype = voxels.dtype
    return tuple(
        jnp.broadcast_to(
            jnp.arange(voxels.shape[axis] + 1, dtype=float_dtype),
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
    float_dtype = jnp.result_type(voxels.dtype, centers.dtype, masses.dtype)
    has_weight, average_radius, max_radius = jax.vmap(
        _radius_statistics_impl, in_axes=(0, 0, 0, None)
    )(voxels, centers, masses, default_radius_multiplier)
    x_samples = (
        jnp.arange(voxels.shape[1] + 1, dtype=float_dtype)[None, :]
        - centers[:, 0, None]
    ) / average_radius[:, None]
    y_samples = (
        jnp.arange(voxels.shape[2] + 1, dtype=float_dtype)[None, :]
        - centers[:, 1, None]
    ) / average_radius[:, None]
    z_samples = (
        jnp.arange(voxels.shape[3] + 1, dtype=float_dtype)[None, :]
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


@partial(jax.jit, static_argnums=(1, 6))
def _calculate_bbox_to_zm_batch(
    bbox_moments: chex.Array,
    max_order: int,
    g_coefficients: chex.Array,
    pqr_indices: chex.Array,
    output_indices: chex.Array,
    clm: chex.Array,
    reduction_strategy: str = "auto",
):
    complex_dtype = jnp.result_type(
        bbox_moments.dtype, g_coefficients.dtype, clm.dtype, jnp.complex64
    )
    complex_moments = jnp.asarray(bbox_moments, dtype=complex_dtype)
    return jax.vmap(
        lambda bbox_moment: _calculate_bbox_moment_2_zm_jax(
            max_order,
            g_coefficients,
            pqr_indices,
            output_indices,
            clm,
            bbox_moment,
            reduction_strategy,
        )
    )(complex_moments)


@partial(jax.jit, static_argnums=(1, 7))
def _calculate_zm_batch(
    voxels: chex.Array,
    max_order: int,
    default_radius_multiplier: float,
    g_coefficients: chex.Array,
    pqr_indices: chex.Array,
    output_indices: chex.Array,
    clm: chex.Array,
    moment_reduction: str = "auto",
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
        moment_reduction,
    )
    return has_weight, scaled, raw


@jax.jit
def _calculate_3dzd_batch(scaled_moments: chex.Array) -> chex.Array:
    return jax.vmap(_get_3dzd_121_descriptor_jax)(scaled_moments)


@partial(jax.jit, static_argnums=(1,))
def _calculate_ab_candidates_batch(raw_moments: chex.Array, target_order: int):
    return jax.vmap(
        lambda raw: calculate_ab_rotation_candidates(raw, target_order, "input")
    )(
        raw_moments
    )


@partial(jax.jit, static_argnums=(1, 2))
def _calculate_ab_compact_candidates_batch(
    raw_moments: chex.Array,
    target_order: int,
    root_strategy: str = "companion",
):
    return jax.vmap(
        lambda raw: calculate_ab_rotation_compact_candidates(
            raw, target_order, root_strategy, "input"
        )
    )(raw_moments)


@partial(jax.jit, static_argnums=(1, 2))
def _calculate_ab_compact_candidate_group_batch(
    raw_moments: chex.Array,
    target_orders: tuple[int, ...],
    root_strategy: str = "companion",
):
    return jax.vmap(
        lambda raw: calculate_ab_rotation_compact_candidate_group(
            raw, target_orders, root_strategy, "input"
        )
    )(raw_moments)


@partial(jax.jit, static_argnums=(2, 12))
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
    reduction_strategy: str = "auto",
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
            reduction_strategy,
        )
    )(raw_moments, pairs)


@partial(jax.jit, static_argnums=(2, 12))
def _calculate_rotation_flat_batch(
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
    reduction_strategy: str = "auto",
) -> chex.Array:
    """Candidate-parallel rotation prototype with a flattened leading layout.

    The protein and candidate axes are flattened before entering the rotation
    kernel and restored afterwards.  Term order, identity handling and the
    deterministic segmented reduction are therefore identical to the nested
    production implementation.
    """
    protein_count, candidate_count = pairs.shape[:2]
    flat_pairs = pairs.reshape((-1, 2))
    flat_raw = jnp.repeat(raw_moments, candidate_count, axis=0)

    def rotate_one(raw, pair):
        return _calculate_zm_by_ab_rotation_jax(
            raw,
            binomial,
            pair[None, :],
            max_order,
            clm,
            s_id,
            n,
            ell,
            m,
            mu,
            k,
            is_nlm_value,
            reduction_strategy,
        )[0]

    rotated = jax.vmap(rotate_one)(flat_raw, flat_pairs)
    return rotated.reshape((protein_count, candidate_count) + rotated.shape[1:])


@jax.jit
def _calculate_masked_mean_batch(
    rotated: chex.Array, is_valid: chex.Array
) -> chex.Array:
    keep = is_valid[..., None, None, None]
    values = jnp.where(keep, jnp.abs(rotated), 0)
    valid_count = jnp.sum(is_valid, axis=1)[:, None, None, None]
    mean = jnp.sum(values, axis=1) / jnp.maximum(valid_count, 1)
    return jnp.where(valid_count > 0, mean, jnp.nan)


@partial(jax.jit, static_argnums=(1, 3, 12))
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
    reduction_strategy: str = "auto",
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
        reduction_strategy,
    )
    return _calculate_masked_mean_batch(rotated, candidates.is_valid)


@partial(jax.jit, static_argnums=(1, 3, 12, 13))
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
    reduction_strategy: str = "auto",
    candidate_root_strategy: str = "companion",
) -> chex.Array:
    candidates = _calculate_ab_compact_candidates_batch(
        raw_moments, target_order, candidate_root_strategy
    )
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
        reduction_strategy,
    )
    return _calculate_masked_mean_batch(rotated, candidates.is_valid)


@partial(jax.jit, static_argnums=(1, 3, 12, 13))
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
    reduction_strategy: str = "auto",
    flattened_rotation: bool = False,
) -> chex.Array:
    """Fuse same-capacity compact orders into one candidate/rotation executable."""
    candidates = _calculate_ab_compact_candidate_group_batch(
        raw_moments, target_orders, "companion"
    )
    pair_groups = candidates.pairs
    mask_groups = candidates.is_valid

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
                reduction_strategy,
            )
        )(item_pair_groups)

    if flattened_rotation:
        rotated = jnp.stack(
            [
                _calculate_rotation_flat_batch(
                    raw_moments,
                    pair_groups[:, index],
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
                    reduction_strategy,
                )
                for index in range(len(target_orders))
            ],
            axis=1,
        )
    else:
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
    reduction_strategy: str,
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
    if representation in ("full_fixed", "analytic_compact", "companion_compact"):
        function = (
            _calculate_normalized_mean_batch
            if representation == "full_fixed"
            else _calculate_normalized_mean_compact_batch
        )
        return jnp.stack(
            [
                (
                    function(
                        raw,
                        target_order,
                        *arguments,
                        reduction_strategy,
                        (
                            "analytic_odd"
                            if representation == "analytic_compact"
                            else "companion"
                        ),
                    )
                    if representation in ("analytic_compact", "companion_compact")
                    else function(raw, target_order, *arguments, reduction_strategy)
                )
                for target_order in target_orders
            ],
            axis=1,
        )

    by_order = {}
    for parity in (0, 1):
        parity_orders = tuple(order for order in target_orders if order % 2 == parity)
        if parity_orders:
            parity_means = _calculate_normalized_means_parity_batch(
                raw,
                parity_orders,
                *arguments,
                reduction_strategy,
                representation == "companion_compact_grouped_flat",
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


def _resolve_moment_precision(max_order: int, moment_precision: str) -> str:
    if moment_precision not in ("auto", "configured", "mixed", "strict"):
        raise ValueError(
            "moment_precision must be 'auto', 'configured', 'mixed', or 'strict'"
        )
    if moment_precision == "auto":
        return "strict" if max_order >= 20 else "configured"
    return moment_precision


def _uses_mixed_moments(max_order: int, moment_precision: str) -> bool:
    """Retain the legacy predicate for benchmark and compatibility callers."""
    return _resolve_moment_precision(max_order, moment_precision) == "mixed"


def _cast_descriptor_values(
    descriptor: DescriptorVector, dtype: jnp.dtype
) -> DescriptorVector:
    return DescriptorVector(
        values=jnp.asarray(descriptor.values, dtype=dtype),
        is_valid=descriptor.is_valid,
    )


def calculate_descriptor_batch_from_voxels(
    voxels: chex.Array,
    *,
    max_order: int,
    max_target_order: int,
    mode: int,
    default_radius_multiplier: float,
    bbox_to_zm_cache: BBoxToZMCache,
    x64_bbox_to_zm_cache: BBoxToZMCache | None = None,
    rotation_cache: ZMRotationCache,
    x64_rotation_cache: ZMRotationCache | None = None,
    descriptor_cache: DescriptorAssemblyCache,
    normalization_representation: str = "companion_compact_grouped",
    rotation_reduction: str = "auto",
    moment_reduction: str = "auto",
    moment_precision: str = "auto",
    moment_backend: str = "cartesian",
    direct_moment_cache: DirectMomentCache | None = None,
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
        "companion_compact",
        "analytic_compact_parity",
        "companion_compact_grouped",
        "companion_compact_grouped_flat",
    ):
        raise ValueError("unknown normalization representation")
    if rotation_reduction not in ("auto", "scatter", "segmented_scan"):
        raise ValueError("unknown rotation reduction")
    if moment_reduction not in ("auto", "scatter", "segmented_scan"):
        raise ValueError("unknown moment reduction")
    if moment_backend not in ("cartesian", "direct_recurrence"):
        raise ValueError("moment_backend must be 'cartesian' or 'direct_recurrence'")
    if moment_backend == "direct_recurrence":
        if moment_precision in ("mixed", "strict"):
            raise ValueError("direct_recurrence supports only auto/configured float32 moments")
        if direct_moment_cache is None or direct_moment_cache.max_order != max_order:
            raise ValueError("a matching direct_moment_cache is required")
        resolved_precision = "configured"
    else:
        resolved_precision = _resolve_moment_precision(max_order, moment_precision)
    use_mixed_moments = resolved_precision == "mixed"
    use_strict_precision = resolved_precision == "strict"
    if use_mixed_moments or use_strict_precision:
        if x64_bbox_to_zm_cache is None:
            raise ValueError(
                f"{resolved_precision} precision requires x64_bbox_to_zm_cache"
            )
        if x64_bbox_to_zm_cache.max_order != max_order:
            raise ValueError("x64 bbox-to-ZM cache maximum order does not match max_order")
    if use_strict_precision:
        if x64_rotation_cache is None:
            raise ValueError("strict precision requires x64_rotation_cache")
        if x64_rotation_cache.max_order != max_order:
            raise ValueError("x64 rotation cache maximum order does not match max_order")

    packed_voxels = (
        None
        if moment_backend != "direct_recurrence" or isinstance(voxels, jax.core.Tracer)
        else pack_occupied_voxels(voxels)
    )
    voxel_dtype = jnp.float64 if use_strict_precision else _config.FLOAT_DTYPE
    voxel_batch = jnp.asarray(voxels, dtype=voxel_dtype)
    if voxel_batch.ndim != 4 or voxel_batch.shape[0] == 0:
        raise ValueError(
            "voxels must have shape (batch, x, y, z) with a non-empty batch"
        )

    if moment_backend == "direct_recurrence":
        masses, centers, _ = _calculate_bbox_order1_batch(voxel_batch)
        radius = _calculate_radius_and_samples_batch(
            voxel_batch, centers, masses, default_radius_multiplier
        )
        scaled, raw = calculate_direct_moments(
            voxel_batch if packed_voxels is None else packed_voxels,
            radius[3], radius[4], radius[5], direct_moment_cache
        )
    elif use_strict_precision:
        masses, centers, _ = _calculate_bbox_order1_batch(voxel_batch)
        radius = _calculate_radius_and_samples_batch(
            voxel_batch, centers, masses, default_radius_multiplier
        )
        _, _, bbox_moments = _calculate_bbox_max_order_batch(
            voxel_batch, max_order, radius[3], radius[4], radius[5]
        )
        scaled, raw = _calculate_bbox_to_zm_batch(
            bbox_moments,
            max_order,
            x64_bbox_to_zm_cache.g_coefficients,
            x64_bbox_to_zm_cache.pqr_indices,
            x64_bbox_to_zm_cache.output_indices,
            x64_bbox_to_zm_cache.clm,
            "segmented_scan",
        )
    elif use_mixed_moments:
        from .mixed_precision_prototype import (
            calculate_bbox_moments_mixed_prototype,
            calculate_zm_mixed_prototype,
        )

        masses, centers, _ = _calculate_bbox_order1_batch(voxel_batch)
        radius = _calculate_radius_and_samples_batch(
            voxel_batch, centers, masses, default_radius_multiplier
        )
        bbox_moments = calculate_bbox_moments_mixed_prototype(
            voxel_batch,
            max_order,
            radius[3],
            radius[4],
            radius[5],
            "moments_x64",
        )
        scaled, raw = calculate_zm_mixed_prototype(
            bbox_moments,
            max_order,
            bbox_to_zm_cache.g_coefficients,
            bbox_to_zm_cache.pqr_indices,
            bbox_to_zm_cache.output_indices,
            bbox_to_zm_cache.clm,
            x64_bbox_to_zm_cache.g_coefficients,
            x64_bbox_to_zm_cache.clm,
            "moments_x64",
        )
    else:
        _, scaled, raw = _calculate_zm_batch(
            voxel_batch,
            max_order,
            default_radius_multiplier,
            bbox_to_zm_cache.g_coefficients,
            bbox_to_zm_cache.pqr_indices,
            bbox_to_zm_cache.output_indices,
            bbox_to_zm_cache.clm,
            moment_reduction,
        )
    descriptors = _calculate_3dzd_batch(scaled) if mode in (1, 2) else None

    active_rotation_cache = (
        x64_rotation_cache if use_strict_precision else rotation_cache
    )
    active_rotation_reduction = (
        "segmented_scan" if use_strict_precision else rotation_reduction
    )
    if mode in (0, 2):
        means = _calculate_normalization_means(
            raw,
            tuple(range(2, max_target_order + 1)),
            normalization_representation,
            active_rotation_cache,
            active_rotation_reduction,
        )
    else:
        max_n = max_order + 1
        means = jnp.empty(
            (voxel_batch.shape[0], 0, max_n, max_n, max_n),
            dtype=_config.FLOAT_DTYPE,
        )

    if descriptors is None:
        result = _assemble_mean_batch(means, descriptor_cache.moment_indices)
    else:
        result = _assemble_descriptor_batch(
            descriptors,
            means,
            descriptor_cache.descriptor_indices,
            descriptor_cache.moment_indices,
        )
    return _cast_descriptor_values(result, _config.FLOAT_DTYPE)


def calculate_descriptor_batch_staged(
    voxels: chex.Array,
    *,
    max_order: int,
    max_target_order: int,
    mode: int,
    default_radius_multiplier: float,
    bbox_to_zm_cache: BBoxToZMCache,
    x64_bbox_to_zm_cache: BBoxToZMCache | None = None,
    rotation_cache: ZMRotationCache,
    x64_rotation_cache: ZMRotationCache | None = None,
    descriptor_cache: DescriptorAssemblyCache,
    stage_executor: StageExecutor | None = None,
    normalization_representation: str = "companion_compact",
    rotation_reduction: str = "auto",
    moment_reduction: str = "auto",
    moment_precision: str = "auto",
    moment_backend: str = "cartesian",
    direct_moment_cache: DirectMomentCache | None = None,
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
    if rotation_reduction not in ("auto", "scatter", "segmented_scan"):
        raise ValueError("unknown rotation reduction")
    if moment_reduction not in ("auto", "scatter", "segmented_scan"):
        raise ValueError("unknown moment reduction")
    if normalization_representation not in (
        "full_fixed", "analytic_compact", "companion_compact"
    ):
        raise ValueError(
            "staged normalization representation must be 'full_fixed' or "
            "'analytic_compact', or 'companion_compact'"
        )
    if moment_backend not in ("cartesian", "direct_recurrence"):
        raise ValueError("moment_backend must be 'cartesian' or 'direct_recurrence'")
    if moment_backend == "direct_recurrence":
        if moment_precision in ("mixed", "strict"):
            raise ValueError("direct_recurrence supports only auto/configured float32 moments")
        if direct_moment_cache is None or direct_moment_cache.max_order != max_order:
            raise ValueError("a matching direct_moment_cache is required")
        resolved_precision = "configured"
    else:
        resolved_precision = _resolve_moment_precision(max_order, moment_precision)
    use_mixed_moments = resolved_precision == "mixed"
    use_strict_precision = resolved_precision == "strict"
    if use_mixed_moments or use_strict_precision:
        if x64_bbox_to_zm_cache is None:
            raise ValueError(
                f"{resolved_precision} precision requires x64_bbox_to_zm_cache"
            )
        if x64_bbox_to_zm_cache.max_order != max_order:
            raise ValueError("x64 bbox-to-ZM cache maximum order does not match max_order")
    if use_strict_precision:
        if x64_rotation_cache is None:
            raise ValueError("strict precision requires x64_rotation_cache")
        if x64_rotation_cache.max_order != max_order:
            raise ValueError("x64 rotation cache maximum order does not match max_order")

    packed_voxels = (
        None
        if moment_backend != "direct_recurrence" or isinstance(voxels, jax.core.Tracer)
        else pack_occupied_voxels(voxels)
    )
    voxel_dtype = jnp.float64 if use_strict_precision else _config.FLOAT_DTYPE
    voxel_batch = jnp.asarray(voxels, dtype=voxel_dtype)
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
    if moment_backend == "direct_recurrence":
        scaled, raw = execute(
            "direct_moments",
            lambda: calculate_direct_moments(
                voxel_batch if packed_voxels is None else packed_voxels,
                x_samples, y_samples, z_samples, direct_moment_cache
            ),
        )
    elif use_strict_precision:
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
                x64_bbox_to_zm_cache.g_coefficients,
                x64_bbox_to_zm_cache.pqr_indices,
                x64_bbox_to_zm_cache.output_indices,
                x64_bbox_to_zm_cache.clm,
                "segmented_scan",
            ),
        )
    elif use_mixed_moments:
        from .mixed_precision_prototype import (
            calculate_bbox_moments_mixed_prototype,
            calculate_zm_mixed_prototype,
        )

        bbox_moments = execute(
            "bbox_max_order",
            lambda: calculate_bbox_moments_mixed_prototype(
                voxel_batch,
                max_order,
                x_samples,
                y_samples,
                z_samples,
                "moments_x64",
            ),
        )
        scaled, raw = execute(
            "bbox_to_zm",
            lambda: calculate_zm_mixed_prototype(
                bbox_moments,
                max_order,
                bbox_to_zm_cache.g_coefficients,
                bbox_to_zm_cache.pqr_indices,
                bbox_to_zm_cache.output_indices,
                bbox_to_zm_cache.clm,
                x64_bbox_to_zm_cache.g_coefficients,
                x64_bbox_to_zm_cache.clm,
                "moments_x64",
            ),
        )
    else:
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
                moment_reduction,
            ),
        )

    descriptors = (
        execute("descriptor_3dzd", lambda: _calculate_3dzd_batch(scaled))
        if mode in (1, 2)
        else None
    )
    active_rotation_cache = (
        x64_rotation_cache if use_strict_precision else rotation_cache
    )
    active_rotation_reduction = (
        "segmented_scan" if use_strict_precision else rotation_reduction
    )
    candidates_by_order: dict[int, Any] = {}
    means_by_order = []
    if mode in (0, 2):
        for target_order in range(2, max_target_order + 1):
            candidates = execute(
                f"ab_candidates_order_{target_order}",
                lambda target_order=target_order: (
                    _calculate_ab_candidates_batch(raw, target_order)
                    if normalization_representation == "full_fixed"
                    else _calculate_ab_compact_candidates_batch(raw, target_order)
                ),
            )
            candidates_by_order[target_order] = candidates
            rotated = execute(
                f"zm_rotation_order_{target_order}",
                lambda candidates=candidates: _calculate_rotation_batch(
                    raw,
                    candidates.pairs,
                    max_order,
                    active_rotation_cache.binomial,
                    active_rotation_cache.clm,
                    active_rotation_cache.s_id,
                    active_rotation_cache.n,
                    active_rotation_cache.l,
                    active_rotation_cache.m,
                    active_rotation_cache.mu,
                    active_rotation_cache.k,
                    active_rotation_cache.is_nlm_value,
                    active_rotation_reduction,
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
    result = _cast_descriptor_values(result, _config.FLOAT_DTYPE)
    return result, candidates_by_order
