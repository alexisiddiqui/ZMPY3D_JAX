"""Device-native descriptor and score API integration coverage."""

import importlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ZMPY3D_JAX as z
from ZMPY3D_JAX.ZMPY3D_CLI_BatchZM import _prepare_batch_runtime
from ZMPY3D_JAX.lib.batched_descriptor import (
    calculate_descriptor_batch_from_voxels,
    calculate_descriptor_batch_staged,
    pad_voxel_batch,
)
from ZMPY3D_JAX.lib.fill_voxel_by_weight_density04 import (
    fill_voxel_by_weight_density_host,
)


def test_zm_modes_and_batch_return_device_vectors(pdb_files):
    path = pdb_files["6NT5"]
    mode0 = z.ZMPY3D_CLI_ZM(path, 1.0, 6, 5, 0)
    mode1 = z.ZMPY3D_CLI_ZM(path, 1.0, 6, 5, 1)
    mode2 = z.ZMPY3D_CLI_ZM(path, 1.0, 6, 5, 2)

    assert isinstance(mode0, z.DescriptorVector)
    assert isinstance(mode0.values, jax.Array)
    assert mode0.values.shape == (200,)
    assert mode1.values.shape == (16,)
    assert mode2.values.shape == (216,)
    assert bool(jnp.all(mode0.is_valid))
    assert bool(jnp.all(mode1.is_valid))
    assert bool(jnp.all(mode2.is_valid))
    np.testing.assert_allclose(
        mode2.values, jnp.concatenate((mode1.values, mode0.values)), rtol=0, atol=0
    )

    batch = z.ZMPY3D_CLI_BatchZM([path], 1.0, 6, 5, 2)
    assert isinstance(batch, z.DescriptorVector)
    assert batch.values.shape == (1, 216)
    assert batch.is_valid.shape == (1, 216)
    np.testing.assert_allclose(batch.values[0], mode2.values, rtol=1e-3, atol=5e-4)
    np.testing.assert_array_equal(batch.is_valid[0], mode2.is_valid)


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_mixed_device_batch_matches_single_descriptors(pdb_files, mode):
    paths = [pdb_files["6NT5"], pdb_files["6NT6"]]
    expected = [z.ZMPY3D_CLI_ZM(path, 1.0, 6, 5, mode) for path in paths]
    actual = z.ZMPY3D_CLI_BatchZM(paths, 1.0, 6, 5, mode, BatchSize=2)

    assert isinstance(actual.values, jax.Array)
    for index, item in enumerate(expected):
        np.testing.assert_allclose(
            actual.values[index], item.values, rtol=1e-3, atol=5e-4, equal_nan=True
        )
        np.testing.assert_array_equal(actual.is_valid[index], item.is_valid)


def test_batch_chunks_preserve_order_and_handle_partial_chunk(pdb_files):
    paths = [pdb_files["6NT6"], pdb_files["6NT5"], pdb_files["6NT6"]]
    expected = [z.ZMPY3D_CLI_ZM(path, 1.0, 6, 5, 1) for path in paths]
    actual = z.ZMPY3D_CLI_BatchZM(paths, 1.0, 6, 5, 1, BatchSize=2)

    assert actual.values.shape == (3, 16)
    for index, item in enumerate(expected):
        np.testing.assert_allclose(
            actual.values[index], item.values, rtol=1e-3, atol=5e-4, equal_nan=True
        )
        np.testing.assert_array_equal(actual.is_valid[index], item.is_valid)


def test_staged_batch_matches_fused_and_reports_candidate_slots(pdb_files):
    runtime = _prepare_batch_runtime(1.0, 6)
    host_voxels = []
    for path in (pdb_files["6NT5"], pdb_files["6NT6"]):
        xyz, residues = z.get_pdb_xyz_ca(path)
        voxel, _ = fill_voxel_by_weight_density_host(
            xyz,
            residues,
            runtime.param["residue_weight_map"],
            1.0,
            runtime.residue_box[1.0],
        )
        host_voxels.append(voxel)
    voxels = jnp.asarray(pad_voxel_batch(host_voxels))
    arguments = {
        "max_order": 6,
        "max_target_order": 5,
        "mode": 2,
        "default_radius_multiplier": runtime.param["default_radius_multiplier"],
        "bbox_to_zm_cache": runtime.bbox_to_zm_cache,
        "rotation_cache": runtime.rotation_cache,
        "descriptor_cache": runtime.descriptor_cache,
    }

    fused = calculate_descriptor_batch_from_voxels(voxels, **arguments)
    compact = calculate_descriptor_batch_from_voxels(
        voxels, **arguments, normalization_representation="analytic_compact"
    )
    parity_fused = calculate_descriptor_batch_from_voxels(
        voxels, **arguments, normalization_representation="analytic_compact_parity"
    )
    grouped_flat = calculate_descriptor_batch_from_voxels(
        voxels,
        **arguments,
        normalization_representation="companion_compact_grouped_flat",
    )
    staged, candidates = calculate_descriptor_batch_staged(voxels, **arguments)

    np.testing.assert_allclose(
        staged.values, fused.values, rtol=2e-6, atol=5e-6, equal_nan=True
    )
    np.testing.assert_array_equal(staged.is_valid, fused.is_valid)
    for alternative in (compact, parity_fused, grouped_flat):
        np.testing.assert_allclose(
            alternative.values, fused.values, rtol=1e-5, atol=1e-5, equal_nan=True
        )
        np.testing.assert_array_equal(alternative.is_valid, fused.is_valid)
    for order, fixed, valid in ((2, 8, 8), (3, 4, 4), (4, 8, 8), (5, 4, 4)):
        assert candidates[order].pairs.shape == (2, fixed, 2)
        np.testing.assert_array_equal(
            jnp.sum(candidates[order].is_valid, axis=1), [valid, valid]
        )


@pytest.mark.slow
def test_order20_mixed_batch_matches_single_descriptors(pdb_files):
    z.configure_for_scientific_computing(enable_x64=True, platform=None)
    paths = [pdb_files["6NT5"], pdb_files["6NT6"]]
    expected = [z.ZMPY3D_CLI_ZM(path, 1.0, 20, 5, 2) for path in paths]
    actual = z.ZMPY3D_CLI_BatchZM(paths, 1.0, 20, 5, 2, BatchSize=2)

    for index, item in enumerate(expected):
        np.testing.assert_allclose(
            actual.values[index], item.values, rtol=1e-3, atol=5e-4, equal_nan=True
        )
        np.testing.assert_array_equal(actual.is_valid[index], item.is_valid)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_batch_size_must_be_a_positive_integer(pdb_files, batch_size):
    with pytest.raises(ValueError, match="BatchSize must be a positive integer"):
        z.ZMPY3D_CLI_BatchZM([pdb_files["6NT5"]], 1.0, 6, 5, 1, BatchSize=batch_size)


def test_empty_descriptor_batches_have_stable_shapes():
    for mode, width in ((0, 200), (1, 16), (2, 216)):
        result = z.ZMPY3D_CLI_BatchZM([], 1.0, 6, 5, mode)
        assert result.values.shape == (0, width)
        assert result.is_valid.shape == (0, width)

    geo, zm = z.ZMPY3D_CLI_BatchShapeScore([], [], 1.0)
    assert geo.shape == (0,)
    assert zm.shape == (0,)
    assert isinstance(geo, jax.Array)
    assert isinstance(zm, jax.Array)


def test_batch_console_forwards_batch_size(monkeypatch, capsys, tmp_path, pdb_files):
    module = importlib.import_module("ZMPY3D_JAX.ZMPY3D_CLI_BatchZM")
    path_list = tmp_path / "inputs.txt"
    path_list.write_text(pdb_files["6NT5"] + "\n", encoding="utf-8")
    captured = {}

    def fake_batch(*args, **kwargs):
        captured.update(kwargs)
        return z.DescriptorVector(
            values=jnp.ones((1, 1)), is_valid=jnp.ones((1, 1), dtype=bool)
        )

    monkeypatch.setattr(module, "ZMPY3D_CLI_BatchZM", fake_batch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ZMPY3D_CLI_BatchZM",
            str(path_list),
            "1.0",
            "6",
            "5",
            "1",
            "--batch-size",
            "3",
        ],
    )

    module.main()

    assert captured["BatchSize"] == 3
    assert "[1.]" in capsys.readouterr().out


def test_shape_score_stays_on_device_and_batches_match(pdb_files):
    path = pdb_files["6NT5"]
    geo, zm = z.ZMPY3D_CLI_ShapeScore(path, path, 1.0)
    assert isinstance(geo, jax.Array)
    assert isinstance(zm, jax.Array)
    np.testing.assert_allclose((geo, zm), (100.0, 100.0), rtol=0, atol=1e-10)

    batch_geo, batch_zm = z.ZMPY3D_CLI_BatchShapeScore([path], [path], 1.0)
    assert batch_geo.shape == (1,)
    assert batch_zm.shape == (1,)
    np.testing.assert_allclose(batch_geo[0], geo, rtol=0, atol=0)
    np.testing.assert_allclose(batch_zm[0], zm, rtol=0, atol=0)


def test_zm_console_compacts_only_at_output(monkeypatch, capsys, pdb_files):
    module = importlib.import_module("ZMPY3D_JAX.ZMPY3D_CLI_ZM")
    result = z.DescriptorVector(
        values=jnp.array([1.0, jnp.nan, 3.0]),
        is_valid=jnp.array([True, False, True]),
    )
    monkeypatch.setattr(module, "ZMPY3D_CLI_ZM", lambda *args: result)
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["ZMPY3D_CLI_ZM", pdb_files["6NT5"], "1.0", "6", "5", "1"],
    )

    module.main()

    assert capsys.readouterr().out.strip() == "[1. 3.]"
