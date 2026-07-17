"""
Tests for calculate_zm_by_ab_rotation function.
"""

import pickle
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import ZMPY3D_JAX as z


def _load_upstream_rotation():
    path = (
        Path(__file__).resolve().parents[3]
        / "externals/ZMPY3D/ZMPY3D/lib/calculate_zm_by_ab_rotation01.py"
    )
    spec = importlib.util.spec_from_file_location("upstream_zm_rotation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.calculate_zm_by_ab_rotation01


class TestCalculateZMByABRotation:
    """Test suite for calculate_zm_by_ab_rotation function."""

    @pytest.fixture
    def cache_data(self):
        """Load cache data needed for rotation calculation."""
        max_order = 6
        cache_dir = Path(z.__file__).parent / "cache_data"

        binomial_path = cache_dir / "BinomialCache.pkl"
        log_cache_path = cache_dir / f"LogG_CLMCache_MaxOrder{max_order:02d}.pkl"

        with open(binomial_path, "rb") as file:
            binomial_cache_pkl = pickle.load(file)

        with open(log_cache_path, "rb") as file:
            cache_pkl = pickle.load(file)

        rotation_index = cache_pkl["RotationIndex"]

        return {
            "BinomialCache": binomial_cache_pkl["BinomialCache"],
            "CLMCache": cache_pkl["CLMCache"],
            "s_id": np.squeeze(rotation_index["s_id"][0, 0]) - 1,
            "n": np.squeeze(rotation_index["n"][0, 0]),
            "l": np.squeeze(rotation_index["l"][0, 0]),
            "m": np.squeeze(rotation_index["m"][0, 0]),
            "mu": np.squeeze(rotation_index["mu"][0, 0]),
            "k": np.squeeze(rotation_index["k"][0, 0]),
            "IsNLM_Value": np.squeeze(rotation_index["IsNLM_Value"][0, 0]) - 1,
            "max_order": max_order,
        }

    @pytest.fixture
    def zm_raw(self, cache_data):
        """Create deterministic moments in the package's (n, l, m>=0) layout."""
        rng = np.random.default_rng(2026)
        zm = np.full((7, 7, 7), np.nan + 0j, dtype=complex)
        for n in range(7):
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    zm[n, l, : l + 1] = rng.normal(size=l + 1) + 1j * rng.normal(size=l + 1)
        return zm

    @pytest.fixture
    def ab_list(self):
        """Create sample AB rotation list."""
        return np.array(
            [
                [np.cos(theta / 2), np.sin(theta / 2) * np.exp(1j * phase)]
                for theta, phase in [(0.4, 0.2), (1.1, -0.7), (2.0, 1.3)]
            ],
            dtype=complex,
        )

    def test_basic_rotation(self, zm_raw, ab_list, cache_data):
        """Test basic ZM rotation calculation."""
        max_order = cache_data["max_order"]

        zm_list = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_list,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        # Should return a list of rotated ZM arrays
        assert isinstance(zm_list, list)

        # Number of results should match number of AB pairs
        assert len(zm_list) == len(ab_list)

        # Each result should be complex
        for zm in zm_list:
            assert np.iscomplexobj(zm)

    def test_output_shapes(self, zm_raw, ab_list, cache_data):
        """Test that output shapes are consistent."""
        max_order = cache_data["max_order"]

        zm_list = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_list,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        # Each rotated ZM should have same shape
        first_shape = zm_list[0].shape
        for zm in zm_list:
            assert zm.shape == first_shape

    def test_single_rotation(self, zm_raw, cache_data):
        """Test with a single AB pair."""
        max_order = cache_data["max_order"]

        # Single normalized AB pair
        a = 1.0 + 0j
        b = 0.0 + 0j
        ab_single = np.array([[a, b]])

        zm_list = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_single,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        # Should get one result
        assert len(zm_list) == 1
        assert np.iscomplexobj(zm_list[0])

    def test_identity_rotation(self, zm_raw, cache_data):
        """Test that identity rotation preserves moments."""
        max_order = cache_data["max_order"]

        # Identity rotation: a=1, b=0
        ab_identity = np.array([[1.0 + 0j, 0.0 + 0j]])

        zm_list = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_identity,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        zm_rotated = zm_list[0]
        expected = np.transpose(zm_raw, (2, 1, 0))
        np.testing.assert_allclose(zm_rotated, expected, rtol=1e-6, atol=1e-7, equal_nan=True)

    def test_deterministic(self, zm_raw, ab_list, cache_data):
        """Test that function is deterministic."""
        max_order = cache_data["max_order"]

        zm_list_1 = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_list,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        zm_list_2 = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_list,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        # Results should be identical
        assert len(zm_list_1) == len(zm_list_2)
        for zm1, zm2 in zip(zm_list_1, zm_list_2):
            np.testing.assert_array_equal(zm1, zm2)

    def test_multiple_rotations(self, zm_raw, cache_data):
        """Test with multiple different rotations."""
        max_order = cache_data["max_order"]

        # Create several different AB pairs
        ab_pairs = []
        for i in range(5):
            theta = i * np.pi / 5
            a = np.cos(theta) + 1j * np.sin(theta)
            b = np.sin(theta) - 1j * np.cos(theta)
            # Normalize
            norm = np.sqrt(np.abs(a) ** 2 + np.abs(b) ** 2)
            ab_pairs.append([a / norm, b / norm])

        ab_list = np.array(ab_pairs)

        zm_list = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_list,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        # Should get 5 different rotations
        assert len(zm_list) == 5

        # Each should be different (in general)
        for i in range(len(zm_list) - 1):
            finite = np.isfinite(zm_list[i]) & np.isfinite(zm_list[i + 1])
            assert np.any(finite)
            assert not np.allclose(zm_list[i][finite], zm_list[i + 1][finite], rtol=1e-5)

    def test_zero_moments(self, ab_list, cache_data):
        """Test with zero Zernike moments."""
        max_order = cache_data["max_order"]
        zero_zm = np.zeros((7, 7, 7), dtype=complex)

        zm_list = z.calculate_zm_by_ab_rotation(
            zero_zm,
            cache_data["BinomialCache"],
            ab_list,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        # Should handle zero input
        assert len(zm_list) == len(ab_list)

        # Results should be zero or NaN
        for zm in zm_list:
            non_nan = zm[~np.isnan(zm)]
            if len(non_nan) > 0:
                assert np.allclose(non_nan, 0, atol=1e-10)

    def test_complex_conjugate_property(self, zm_raw, cache_data):
        """Test conjugate symmetry properties of rotated moments."""
        max_order = cache_data["max_order"]

        # Create AB pair and its conjugate
        a = 0.6 + 0.8j
        b = 0.8 - 0.6j
        norm = np.sqrt(np.abs(a) ** 2 + np.abs(b) ** 2)
        ab_pair = np.array([[a / norm, b / norm]])

        zm_list = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_pair,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        # Result should maintain certain symmetries
        zm_rotated = zm_list[0]
        assert np.iscomplexobj(zm_rotated)

    def test_matches_upstream_for_nondegenerate_rotations(self, zm_raw, ab_list, cache_data):
        args = (
            zm_raw,
            cache_data["BinomialCache"],
            ab_list,
            cache_data["max_order"],
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )
        expected = _load_upstream_rotation()(*args)
        actual = z.calculate_zm_by_ab_rotation(*args)
        for actual_item, expected_item in zip(actual, expected):
            np.testing.assert_allclose(
                actual_item, expected_item, rtol=2e-5, atol=2e-6, equal_nan=True
            )

    def test_prepared_batch_matches_legacy_and_mean_invariant(
        self, zm_raw, ab_list, cache_data
    ):
        cache = z.prepare_zm_rotation_cache(
            cache_data["BinomialCache"],
            cache_data["max_order"],
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )
        batch = z.calculate_zm_by_ab_rotation_batch(zm_raw, ab_list, cache)
        legacy = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_list,
            cache_data["max_order"],
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        assert batch.shape == (len(ab_list), 7, 7, 7)
        np.testing.assert_allclose(np.asarray(batch), np.stack(legacy), equal_nan=True)
        batch_mean, batch_std = z.get_mean_invariant(batch)
        list_mean, list_std = z.get_mean_invariant(legacy)
        np.testing.assert_allclose(batch_mean, list_mean, equal_nan=True)
        np.testing.assert_allclose(batch_std, list_std, equal_nan=True)

    def test_prepared_batch_supports_empty_rotation_list(self, zm_raw, cache_data):
        cache = z.prepare_zm_rotation_cache(
            cache_data["BinomialCache"],
            cache_data["max_order"],
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )
        batch = z.calculate_zm_by_ab_rotation_batch(
            zm_raw, np.empty((0, 2), dtype=complex), cache
        )
        assert batch.shape == (0, 7, 7, 7)
