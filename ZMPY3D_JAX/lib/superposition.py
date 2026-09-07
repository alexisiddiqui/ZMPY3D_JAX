"""Shared fixed-shape, device-native superposition internals."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

import ZMPY3D_JAX.config as _config

from .calculate_ab_candidates_jax import (
    ABRotationCandidates,
    calculate_ab_rotation_all_candidates,
)
from .calculate_bbox_moment06 import calculate_bbox_moment06
from .calculate_bbox_moment_2_zm05 import (
    BBoxToZMCache,
    calculate_bbox_moment_2_zm_cached,
    prepare_bbox_to_zm_cache,
)
from .calculate_molecular_radius03 import (
    calculate_molecular_radius_and_bbox_samples,
)
from .calculate_zm_by_ab_rotation01 import (
    ZMRotationCache,
    calculate_zm_by_ab_rotation_batch,
    prepare_zm_rotation_cache,
)
from .fill_voxel_by_weight_density04 import fill_voxel_by_weight_density04
from .get_global_parameter02 import get_global_parameter02
from .get_pdb_xyz_ca02 import get_pdb_xyz_ca02
from .get_residue_gaussian_density_cache02 import (
    get_residue_gaussian_density_cache02,
)
from .get_transform_matrix_from_ab_list02 import get_transform_matrix_from_ab_list02


MAX_ORDER = 6
GRID_WIDTH = 1.0
TARGET_ORDERS = (2, 3, 4, 5, 6)


class SuperpositionRuntime(NamedTuple):
    """Host and device constants shared by one superposition invocation."""

    residue_weight_map: dict[str, float]
    default_radius_multiplier: float
    residue_box: dict[str, np.ndarray]
    rotation_cache: ZMRotationCache
    bbox_to_zm_cache: BBoxToZMCache
    rotation_feature_indices: chex.Array


class SuperpositionFeatures(NamedTuple):
    """Fixed candidate features for one structure."""

    center_scaled: chex.Array
    pairs: chex.Array
    is_valid: chex.Array
    values: chex.Array


class SuperpositionMatch(NamedTuple):
    """Selected candidate pair and the resulting A-to-B transformation."""

    matrix: chex.Array
    index_a: chex.Array
    index_b: chex.Array
    similarity: chex.Array
    is_valid: chex.Array


class SuperpositionSelection(NamedTuple):
    """Best fixed candidate indices before transform construction."""

    index_a: chex.Array
    index_b: chex.Array
    similarity: chex.Array
    is_valid: chex.Array


def prepare_superposition_runtime() -> SuperpositionRuntime:
    """Load the fixed order-6 superposition constants once."""
    param = get_global_parameter02()
    residue_boxes = get_residue_gaussian_density_cache02(param)
    cache_dir = Path(__file__).resolve().parents[1] / "cache_data"
    with (cache_dir / "BinomialCache.pkl").open("rb") as handle:
        binomial = pickle.load(handle)["BinomialCache"]
    with (cache_dir / f"LogG_CLMCache_MaxOrder{MAX_ORDER:02d}.pkl").open(
        "rb"
    ) as handle:
        cache: dict[str, Any] = pickle.load(handle)

    rotation_index = cache["RotationIndex"]
    rotation_cache = prepare_zm_rotation_cache(
        binomial,
        MAX_ORDER,
        cache["CLMCache"],
        np.squeeze(rotation_index["s_id"][0, 0]) - 1,
        np.squeeze(rotation_index["n"][0, 0]),
        np.squeeze(rotation_index["l"][0, 0]),
        np.squeeze(rotation_index["m"][0, 0]),
        np.squeeze(rotation_index["mu"][0, 0]),
        np.squeeze(rotation_index["k"][0, 0]),
        np.squeeze(rotation_index["IsNLM_Value"][0, 0]) - 1,
    )
    bbox_to_zm_cache = prepare_bbox_to_zm_cache(
        MAX_ORDER,
        cache["GCache_complex"],
        cache["GCache_pqr_linear"],
        cache["GCache_complex_index"],
        cache["CLMCache3D"],
    )
    return SuperpositionRuntime(
        residue_weight_map=param["residue_weight_map"],
        default_radius_multiplier=param["default_radius_multiplier"],
        residue_box=residue_boxes[GRID_WIDTH],
        rotation_cache=rotation_cache,
        bbox_to_zm_cache=bbox_to_zm_cache,
        rotation_feature_indices=jnp.sort(rotation_cache.is_nlm_value),
    )


def calculate_structure_moments(
    pdb_file_name: str, runtime: SuperpositionRuntime
) -> tuple[chex.Array, chex.Array]:
    """Convert one legacy PDB CA trace into its center and raw moments."""
    xyz, residue_names = get_pdb_xyz_ca02(pdb_file_name)
    voxel, corner = fill_voxel_by_weight_density04(
        xyz,
        residue_names,
        runtime.residue_weight_map,
        GRID_WIDTH,
        runtime.residue_box,
    )
    samples = {
        "X_sample": jnp.arange(voxel.shape[0] + 1, dtype=_config.FLOAT_DTYPE),
        "Y_sample": jnp.arange(voxel.shape[1] + 1, dtype=_config.FLOAT_DTYPE),
        "Z_sample": jnp.arange(voxel.shape[2] + 1, dtype=_config.FLOAT_DTYPE),
    }
    mass, center, _ = calculate_bbox_moment06(voxel, 1, samples)
    _, _, sphere_samples = calculate_molecular_radius_and_bbox_samples(
        voxel,
        center,
        mass,
        runtime.default_radius_multiplier,
    )
    center_scaled = center * GRID_WIDTH + corner
    _, _, sphere_moment = calculate_bbox_moment06(
        voxel, MAX_ORDER, sphere_samples
    )
    _, raw_moment = calculate_bbox_moment_2_zm_cached(
        sphere_moment, runtime.bbox_to_zm_cache
    )
    return center_scaled, raw_moment


def calculate_superposition_candidates(raw_moment: chex.Array) -> ABRotationCandidates:
    """Concatenate all order-2 through order-6 candidates without compaction."""
    candidates = [
        calculate_ab_rotation_all_candidates(raw_moment, order)
        for order in TARGET_ORDERS
    ]
    pairs = jnp.concatenate(
        [candidate.pairs.reshape((-1, 2)) for candidate in candidates], axis=0
    )
    is_valid = jnp.concatenate(
        [candidate.is_valid.reshape(-1) for candidate in candidates], axis=0
    )
    identity = jnp.asarray((1.0 + 0.0j, 0.0 + 0.0j), dtype=pairs.dtype)
    safe_pairs = jnp.where(is_valid[:, None], pairs, identity[None, :])
    return ABRotationCandidates(safe_pairs, is_valid)


def calculate_superposition_rotations(
    raw_moment: chex.Array,
    candidates: ABRotationCandidates,
    runtime: SuperpositionRuntime,
) -> chex.Array:
    """Evaluate every fixed candidate; invalid slots contain identity rotations."""
    return calculate_zm_by_ab_rotation_batch(
        raw_moment, candidates.pairs, runtime.rotation_cache
    )


@jax.jit
def assemble_superposition_features(
    center_scaled: chex.Array,
    candidates: ABRotationCandidates,
    rotated_moments: chex.Array,
    rotation_feature_indices: chex.Array,
) -> SuperpositionFeatures:
    """Gather structural ``(n, l, m)`` slots into candidate feature columns."""
    candidate_major = jnp.transpose(rotated_moments, (0, 3, 2, 1)).reshape(
        (rotated_moments.shape[0], -1)
    )
    values = candidate_major[:, rotation_feature_indices].T
    return SuperpositionFeatures(
        center_scaled=jnp.asarray(center_scaled, dtype=_config.FLOAT_DTYPE),
        pairs=candidates.pairs,
        is_valid=candidates.is_valid,
        values=values,
    )


def calculate_superposition_features(
    center_scaled: chex.Array,
    raw_moment: chex.Array,
    runtime: SuperpositionRuntime,
) -> SuperpositionFeatures:
    """Build all fixed-shape candidate features for one structure."""
    candidates = calculate_superposition_candidates(raw_moment)
    rotated = calculate_superposition_rotations(raw_moment, candidates, runtime)
    return assemble_superposition_features(
        center_scaled, candidates, rotated, runtime.rotation_feature_indices
    )


@jax.jit
def select_superposition_features(
    features_a: SuperpositionFeatures,
    features_b: SuperpositionFeatures,
) -> SuperpositionSelection:
    """Select the first row-major maximum among valid candidate pairs."""
    similarity = jnp.abs(features_a.values.conj().T @ features_b.values)
    valid_pairs = features_a.is_valid[:, None] & features_b.is_valid[None, :]
    masked_similarity = jnp.where(valid_pairs, similarity, -jnp.inf)
    flat_index = jnp.argmax(masked_similarity.reshape(-1))
    index_a, index_b = jnp.unravel_index(flat_index, masked_similarity.shape)
    has_valid_pair = jnp.any(valid_pairs)
    return SuperpositionSelection(
        index_a=index_a,
        index_b=index_b,
        similarity=masked_similarity[index_a, index_b],
        is_valid=has_valid_pair,
    )


@jax.jit
def solve_superposition_transform(
    features_a: SuperpositionFeatures,
    features_b: SuperpositionFeatures,
    selection: SuperpositionSelection,
) -> chex.Array:
    """Construct and solve the selected A-to-B homogeneous transform."""

    def solve_transform(_):
        pair_a = features_a.pairs[selection.index_a]
        pair_b = features_b.pairs[selection.index_b]
        rotation_a = get_transform_matrix_from_ab_list02(
            pair_a[0], pair_a[1], features_a.center_scaled
        )
        rotation_b = get_transform_matrix_from_ab_list02(
            pair_b[0], pair_b[1], features_b.center_scaled
        )
        return jnp.linalg.solve(rotation_b, rotation_a)

    def invalid_transform(_):
        return jnp.full((4, 4), jnp.nan, dtype=_config.FLOAT_DTYPE)

    return jax.lax.cond(
        selection.is_valid, solve_transform, invalid_transform, operand=None
    )


@jax.jit
def match_superposition_features(
    features_a: SuperpositionFeatures,
    features_b: SuperpositionFeatures,
) -> SuperpositionMatch:
    """Select the best valid candidate pair and solve the A-to-B transform."""
    selection = select_superposition_features(features_a, features_b)
    matrix = solve_superposition_transform(features_a, features_b, selection)
    return SuperpositionMatch(
        matrix=matrix,
        index_a=selection.index_a,
        index_b=selection.index_b,
        similarity=selection.similarity,
        is_valid=selection.is_valid,
    )


def calculate_pdb_superposition(
    pdb_file_name_a: str,
    pdb_file_name_b: str,
    runtime: SuperpositionRuntime,
) -> SuperpositionMatch:
    """Run the shared CA/PDB superposition path for one structure pair."""
    center_a, raw_a = calculate_structure_moments(pdb_file_name_a, runtime)
    center_b, raw_b = calculate_structure_moments(pdb_file_name_b, runtime)
    features_a = calculate_superposition_features(center_a, raw_a, runtime)
    features_b = calculate_superposition_features(center_b, raw_b, runtime)
    return match_superposition_features(features_a, features_b)
