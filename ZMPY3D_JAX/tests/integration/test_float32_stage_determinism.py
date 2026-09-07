"""Isolated repeatability coverage for float32 descriptor pipeline stages."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _assert_tree_identical(actual, expected, stage: str) -> None:
    import jax
    import numpy as np

    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for leaf_index, (actual_leaf, expected_leaf) in enumerate(
        zip(actual_leaves, expected_leaves, strict=True)
    ):
        actual_array = np.asarray(actual_leaf)
        expected_array = np.asarray(expected_leaf)
        if not np.array_equal(actual_array, expected_array, equal_nan=True):
            difference = np.abs(actual_array - expected_array)
            finite = np.isfinite(difference)
            max_difference = float(np.max(difference[finite])) if np.any(finite) else 0.0
            raise AssertionError(
                f"{stage} leaf {leaf_index} is not repeatable; "
                f"max_abs_difference={max_difference:.9g}"
            )


def _run_worker(max_order: int, backend: str) -> None:
    import jax
    import jax.numpy as jnp

    import ZMPY3D_JAX as z

    z.configure_for_scientific_computing(enable_x64=False, platform=backend)

    from ZMPY3D_JAX.ZMPY3D_CLI_BatchZM import (
        _prepare_batch_runtime,
        _prepare_descriptor_runner,
    )
    from ZMPY3D_JAX.lib.batched_descriptor import (
        _calculate_bbox_max_order_batch,
        _calculate_bbox_order1_batch,
        _calculate_bbox_to_zm_batch,
        _calculate_radius_and_samples_batch,
        calculate_descriptor_batch_from_voxels,
        pad_voxel_batch,
    )
    from ZMPY3D_JAX.lib.fill_voxel_by_weight_density04 import (
        fill_voxel_by_weight_density_host,
    )

    runtime = _prepare_batch_runtime(1.0, max_order)
    repo_root = Path(__file__).resolve().parents[3]
    host_voxels = []
    for name in ("6NT5.pdb", "6NT6.pdb"):
        xyz, residues = z.get_pdb_xyz_ca(str(repo_root / name))
        voxel, _ = fill_voxel_by_weight_density_host(
            xyz,
            residues,
            runtime.param["residue_weight_map"],
            1.0,
            runtime.residue_box[1.0],
        )
        host_voxels.append(voxel)

    voxels = jnp.asarray(pad_voxel_batch(host_voxels), dtype=jnp.float32)
    order1 = _calculate_bbox_order1_batch(voxels)
    radius = _calculate_radius_and_samples_batch(
        voxels,
        order1[1],
        order1[0],
        runtime.param["default_radius_multiplier"],
    )
    bbox = _calculate_bbox_max_order_batch(
        voxels, max_order, radius[3], radius[4], radius[5]
    )
    zm = _calculate_bbox_to_zm_batch(
        bbox[2],
        max_order,
        runtime.bbox_to_zm_cache.g_coefficients,
        runtime.bbox_to_zm_cache.pqr_indices,
        runtime.bbox_to_zm_cache.output_indices,
        runtime.bbox_to_zm_cache.clm,
        "auto",
    )
    jax.block_until_ready((order1, radius, bbox, zm))
    compiled_descriptor = _prepare_descriptor_runner(
        max_order=max_order,
        max_target_order=5,
        mode=2,
        runtime=runtime,
    )

    stages = {
        "bbox_order1": lambda: _calculate_bbox_order1_batch(voxels),
        "radius_and_samples": lambda: _calculate_radius_and_samples_batch(
            voxels,
            order1[1],
            order1[0],
            runtime.param["default_radius_multiplier"],
        ),
        "bbox_max_order": lambda: _calculate_bbox_max_order_batch(
            voxels, max_order, radius[3], radius[4], radius[5]
        ),
        "bbox_to_zm": lambda: _calculate_bbox_to_zm_batch(
            bbox[2],
            max_order,
            runtime.bbox_to_zm_cache.g_coefficients,
            runtime.bbox_to_zm_cache.pqr_indices,
            runtime.bbox_to_zm_cache.output_indices,
            runtime.bbox_to_zm_cache.clm,
            "auto",
        ),
        "descriptor_pipeline": lambda: calculate_descriptor_batch_from_voxels(
            voxels,
            max_order=max_order,
            max_target_order=5,
            mode=2,
            default_radius_multiplier=runtime.param[
                "default_radius_multiplier"
            ],
            bbox_to_zm_cache=runtime.bbox_to_zm_cache,
            x64_bbox_to_zm_cache=runtime.x64_bbox_to_zm_cache,
            rotation_cache=runtime.rotation_cache,
            x64_rotation_cache=runtime.x64_rotation_cache,
            descriptor_cache=runtime.descriptor_cache,
        ),
        "compiled_descriptor_pipeline": lambda: compiled_descriptor(voxels),
    }
    for stage, calculate in stages.items():
        baseline = calculate()
        jax.block_until_ready(baseline)
        for _ in range(4):
            repeated = calculate()
            jax.block_until_ready(repeated)
            _assert_tree_identical(repeated, baseline, stage)


def _run_isolated(max_order: int) -> None:
    backend = os.getenv("ZMPY3D_FLOAT32_REGRESSION_BACKEND", "cpu").lower()
    if backend not in ("cpu", "gpu"):
        raise ValueError(
            "ZMPY3D_FLOAT32_REGRESSION_BACKEND must be 'cpu' or 'gpu'"
        )
    subprocess.run(
        [sys.executable, __file__, "--worker", str(max_order), backend],
        check=True,
    )


def test_float32_order6_stages_are_repeatable() -> None:
    _run_isolated(6)


@pytest.mark.slow
def test_float32_order20_stages_are_repeatable() -> None:
    _run_isolated(20)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("usage: test_float32_stage_determinism.py --worker ORDER BACKEND")
    _run_worker(int(sys.argv[2]), sys.argv[3])
