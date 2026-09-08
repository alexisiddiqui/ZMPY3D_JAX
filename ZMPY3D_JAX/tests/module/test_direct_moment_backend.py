import pickle
from pathlib import Path

import numpy as np
import pytest

import ZMPY3D_JAX as z
from ZMPY3D_JAX.lib.calculate_bbox_moment06 import _calculate_bbox_moment_jax


def _cache(order):
    path = Path(z.__file__).parent / "cache_data" / f"LogG_CLMCache_MaxOrder{order:02d}.pkl"
    with path.open("rb") as handle:
        data = pickle.load(handle)
    legacy = z.prepare_bbox_to_zm_cache(order, data["GCache_complex"], data["GCache_pqr_linear"], data["GCache_complex_index"], data["CLMCache3D"])
    return data, legacy, z.prepare_direct_moment_cache(order, data["CLMCache3D"])


@pytest.mark.parametrize("order", [6, pytest.param(20, marks=pytest.mark.slow)])
def test_exact_cell_quadrature_matches_legacy(order):
    z.configure_for_scientific_computing(enable_x64=True, platform="cpu")
    _, legacy_cache, direct_cache = _cache(order)
    rng = np.random.default_rng(731)
    voxels = rng.uniform(size=(1, 3, 2, 4))
    edges = (
        np.linspace(-.8, .7, 4)[None, :],
        np.linspace(-.4, .9, 3)[None, :],
        np.linspace(-.6, 1.2, 5)[None, :],
    )
    _, _, bbox = _calculate_bbox_moment_jax(voxels[0], order, *(edge[0] for edge in edges))
    expected_scaled, expected_raw = z.calculate_bbox_moment_2_zm_cached(bbox, legacy_cache, reduction_strategy="segmented_scan")
    actual_scaled, actual_raw = z.calculate_direct_moments(voxels, *edges, direct_cache, tile_size=5)
    mask = np.isfinite(np.asarray(expected_raw))
    np.testing.assert_allclose(np.asarray(actual_raw[0])[mask], np.asarray(expected_raw)[mask], rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(np.asarray(actual_scaled[0])[mask], np.asarray(expected_scaled)[mask], rtol=2e-9, atol=2e-10)
    assert np.array_equal(np.isnan(actual_raw[0]), np.isnan(expected_raw))


def test_packed_padding_has_zero_contribution():
    data, _, cache = _cache(6)
    voxels = np.zeros((2, 2, 2, 2)); voxels[0, 0, 0, 0] = 1; voxels[1, 1, 1, 1] = 2
    edges = np.arange(3, dtype=float)[None, :].repeat(2, axis=0)
    dense = z.calculate_direct_moments(voxels, edges, edges, edges, cache)
    from ZMPY3D_JAX.lib.direct_moment_backend import pack_occupied_voxels
    packed = z.calculate_direct_moments(pack_occupied_voxels(voxels), edges, edges, edges, cache)
    np.testing.assert_array_equal(dense[0], packed[0])
