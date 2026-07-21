"""Isolated float32 parity coverage for normalization representations."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPRESENTATIONS = (
    "full_fixed",
    "analytic_compact",
    "analytic_compact_parity",
)


def _run_worker(max_order: int, backend: str) -> None:
    import numpy as np
    import jax.numpy as jnp

    import ZMPY3D_JAX as z

    z.configure_for_scientific_computing(enable_x64=False, platform=backend)

    from ZMPY3D_JAX.ZMPY3D_CLI_BatchZM import _prepare_batch_runtime
    from ZMPY3D_JAX.lib.batched_descriptor import (
        _assemble_mean_batch,
        _calculate_ab_candidates_batch,
        _calculate_ab_compact_candidates_batch,
        _calculate_normalization_means,
        _calculate_zm_batch,
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
    raw = _calculate_zm_batch(
        voxels,
        max_order,
        runtime.param["default_radius_multiplier"],
        runtime.bbox_to_zm_cache.g_coefficients,
        runtime.bbox_to_zm_cache.pqr_indices,
        runtime.bbox_to_zm_cache.output_indices,
        runtime.bbox_to_zm_cache.clm,
    )[2]
    raw.block_until_ready()

    target_orders = (2, 3, 4, 5)

    def calculate(representation: str, reduction_strategy: str = "auto"):
        means = _calculate_normalization_means(
            raw,
            target_orders,
            representation,
            runtime.rotation_cache,
            reduction_strategy,
        )
        return _assemble_mean_batch(
            means, runtime.descriptor_cache.moment_indices
        )

    results = {name: calculate(name) for name in REPRESENTATIONS}
    for result in results.values():
        result.values.block_until_ready()

    baseline = results["full_fixed"]
    assert baseline.values.dtype == jnp.float32
    explicit_segmented = calculate(
        "analytic_compact", reduction_strategy="segmented_scan"
    )
    np.testing.assert_array_equal(
        explicit_segmented.values, results["analytic_compact"].values
    )
    for representation in REPRESENTATIONS[1:]:
        result = results[representation]
        np.testing.assert_array_equal(result.is_valid, baseline.is_valid)
        np.testing.assert_array_equal(
            np.isnan(result.values), np.isnan(baseline.values)
        )

        if max_order == 6 and backend == "cpu":
            rtol, atol = 1e-4, 5e-5
        elif max_order == 6:
            rtol, atol = 1e-3, 1e-4
        else:
            rtol, atol = 2e-3, 2e-3
        np.testing.assert_allclose(
            result.values, baseline.values, rtol=rtol, atol=atol, equal_nan=True
        )
        if max_order == 20 and backend == "gpu":
            difference = np.abs(np.asarray(result.values - baseline.values))
            scale = np.maximum(1, np.abs(np.asarray(baseline.values)))
            assert float(np.max(difference / scale)) < 0.01

    for target_order in target_orders:
        full = _calculate_ab_candidates_batch(raw, target_order)
        compact = _calculate_ab_compact_candidates_batch(raw, target_order)
        np.testing.assert_array_equal(
            jnp.sum(full.is_valid, axis=1), jnp.sum(compact.is_valid, axis=1)
        )
        assert full.pairs.shape[1] == 2 * compact.pairs.shape[1]

    repeated = [np.asarray(calculate("analytic_compact").values) for _ in range(5)]
    for result in repeated[1:]:
        np.testing.assert_array_equal(result, repeated[0])


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


def test_float32_order6_normalization_representations() -> None:
    _run_isolated(6)


@pytest.mark.slow
def test_float32_order20_normalization_representations() -> None:
    _run_isolated(20)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("usage: test_float32_normalization_regression.py --worker ORDER BACKEND")
    _run_worker(int(sys.argv[2]), sys.argv[3])
