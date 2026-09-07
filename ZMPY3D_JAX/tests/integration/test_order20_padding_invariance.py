"""Regression coverage for heterogeneous order-20 padding invariance."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _run_worker() -> None:
    import jax.numpy as jnp
    import numpy as np

    import ZMPY3D_JAX as z

    z.configure_for_scientific_computing(enable_x64=False, platform="cpu")

    from ZMPY3D_JAX.ZMPY3D_CLI_BatchZM import _prepare_batch_runtime
    from ZMPY3D_JAX.lib.batched_descriptor import (
        calculate_descriptor_batch_from_voxels,
    )
    from ZMPY3D_JAX.lib.fill_voxel_by_weight_density04 import (
        fill_voxel_by_weight_density_host,
    )

    fixture = Path(__file__).resolve().parents[1] / "data" / "9j1r_ca_regression.pdb"
    runtime = _prepare_batch_runtime(1.0, 20)
    xyz, residues = z.get_pdb_xyz_ca(str(fixture))
    voxel, _ = fill_voxel_by_weight_density_host(
        xyz,
        residues,
        runtime.param["residue_weight_map"],
        1.0,
        runtime.residue_box[1.0],
    )
    padded = np.zeros((1, 106, 123, 142), dtype=np.float32)
    padded[0, : voxel.shape[0], : voxel.shape[1], : voxel.shape[2]] = voxel

    def calculate(values, precision="auto"):
        return calculate_descriptor_batch_from_voxels(
            jnp.asarray(values),
            max_order=20,
            max_target_order=5,
            mode=2,
            default_radius_multiplier=runtime.param["default_radius_multiplier"],
            bbox_to_zm_cache=runtime.bbox_to_zm_cache,
            x64_bbox_to_zm_cache=runtime.x64_bbox_to_zm_cache,
            rotation_cache=runtime.rotation_cache,
            x64_rotation_cache=runtime.x64_rotation_cache,
            descriptor_cache=runtime.descriptor_cache,
            moment_precision=precision,
        )

    native = calculate(voxel[None, ...])
    padded_result = calculate(padded)
    explicit_strict = calculate(padded, "strict")

    assert padded_result.values.dtype == jnp.float32
    np.testing.assert_array_equal(padded_result.is_valid, native.is_valid)
    np.testing.assert_array_equal(explicit_strict.is_valid, padded_result.is_valid)
    assert bool(np.all(np.isfinite(np.asarray(padded_result.values[padded_result.is_valid]))))
    np.testing.assert_allclose(
        padded_result.values,
        native.values,
        rtol=2e-5,
        atol=5e-6,
        equal_nan=True,
    )
    np.testing.assert_array_equal(explicit_strict.values, padded_result.values)


@pytest.mark.slow
def test_order20_auto_precision_is_padding_invariant() -> None:
    subprocess.run([sys.executable, __file__, "--worker"], check=True)


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("usage: test_order20_padding_invariance.py --worker")
    _run_worker()
