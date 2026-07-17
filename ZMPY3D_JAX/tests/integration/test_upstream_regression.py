"""End-to-end numerical regression coverage against the original NumPy package."""

from __future__ import annotations

import numpy as np
import pytest

import ZMPY3D_JAX as z

z.configure_for_scientific_computing(enable_x64=True, platform="cpu")

from ZMPY3D_JAX.tests.utils.upstream_regression import (
    canonicalize_ab_pairs,
    jax_setup,
    pdb_input,
    run_pipeline,
    synthetic_inputs,
    upstream_functions,
    upstream_setup,
)


def _assert_array_parity(actual, expected, *, rtol: float, atol: float) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    finite = np.isfinite(actual) & np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=rtol, atol=atol)


def _assert_pipeline_parity(actual: dict, expected: dict) -> None:
    tolerances = {
        "voxel": (5e-5, 5e-6),
        "corner": (1e-7, 5e-6),
        "mass": (5e-5, 5e-6),
        "center": (5e-5, 5e-6),
        "bbox_order1": (5e-5, 5e-6),
        "average_radius": (5e-6, 5e-7),
        "max_radius": (5e-6, 5e-7),
        "mass_n": (5e-5, 5e-6),
        "center_n": (5e-5, 5e-6),
        "bbox_order_n": (1e-3, 3e-4),
        "scaled": (1e-3, 3e-4),
        "raw": (1e-3, 3e-4),
        "descriptor": (5e-5, 5e-6),
    }
    for name, (rtol, atol) in tolerances.items():
        _assert_array_parity(actual[name], expected[name], rtol=rtol, atol=atol)

    assert actual["sphere"].keys() == expected["sphere"].keys()
    for key in actual["sphere"]:
        _assert_array_parity(actual["sphere"][key], expected["sphere"][key], rtol=5e-5, atol=5e-6)

    assert actual["candidates"].keys() == expected["candidates"].keys()
    for target_order in actual["candidates"]:
        actual_groups = actual["candidates"][target_order]
        expected_groups = expected["candidates"][target_order]
        assert len(actual_groups) == len(expected_groups)
        for actual_group, expected_group in zip(actual_groups, expected_groups):
            _assert_array_parity(
                canonicalize_ab_pairs(actual_group),
                canonicalize_ab_pairs(expected_group),
                rtol=2e-4,
                atol=2e-5,
            )

    assert len(actual["rotated"]) == len(expected["rotated"])
    for actual_rotation, expected_rotation in zip(actual["rotated"], expected["rotated"]):
        _assert_array_parity(
            actual_rotation,
            expected_rotation,
            rtol=1e-3,
            atol=3e-4,
        )


@pytest.mark.filterwarnings("ignore:numpy.fix is deprecated:DeprecationWarning")
def test_reference_parameters_and_residue_caches_match() -> None:
    actual_params, actual_boxes = jax_setup()
    expected_params, expected_boxes = upstream_setup()

    assert actual_params.keys() == expected_params.keys()
    for key in actual_params:
        if isinstance(actual_params[key], dict):
            assert actual_params[key] == expected_params[key]
        elif isinstance(actual_params[key], list):
            assert actual_params[key] == expected_params[key]
        else:
            np.testing.assert_allclose(actual_params[key], expected_params[key], rtol=0, atol=0)

    for grid_width in (0.25, 0.5, 1.0):
        actual_grid = actual_boxes[grid_width]
        expected_grid = expected_boxes[grid_width]
        assert actual_grid.keys() == expected_grid.keys()
        for residue in actual_grid:
            _assert_array_parity(
                actual_grid[residue], expected_grid[residue], rtol=5e-5, atol=5e-6
            )


@pytest.mark.parametrize("case", synthetic_inputs(), ids=lambda case: case.name)
@pytest.mark.filterwarnings("ignore:numpy.fix is deprecated:DeprecationWarning")
def test_synthetic_pipeline_matches_upstream(case) -> None:
    actual = run_pipeline("jax", case)
    expected = run_pipeline("upstream", case)
    _assert_pipeline_parity(actual, expected)


@pytest.mark.parametrize("pdb_name", ("6NT5", "6NT6"))
@pytest.mark.filterwarnings("ignore:numpy.fix is deprecated:DeprecationWarning")
def test_real_pdb_pipeline_matches_upstream(pdb_name, pdb_files) -> None:
    path = pdb_files[pdb_name]
    actual_xyz, actual_residues = z.get_pdb_xyz_ca(path)
    expected_xyz, expected_residues = upstream_functions()["parse_pdb"](path)
    _assert_array_parity(actual_xyz, expected_xyz, rtol=0, atol=0)
    assert actual_residues == expected_residues

    case = pdb_input(path)
    actual = run_pipeline("jax", case)
    expected = run_pipeline("upstream", case)
    _assert_pipeline_parity(actual, expected)


@pytest.mark.slow
@pytest.mark.parametrize("case_name", ("synthetic", "6NT5"))
@pytest.mark.filterwarnings("ignore:numpy.fix is deprecated:DeprecationWarning")
def test_order20_pipeline_matches_upstream(case_name, pdb_files) -> None:
    if case_name == "synthetic":
        case = synthetic_inputs()[0].with_order(20)
    else:
        case = pdb_input(pdb_files["6NT5"], max_order=20)
    actual = run_pipeline("jax", case)
    expected = run_pipeline("upstream", case)
    _assert_pipeline_parity(actual, expected)
