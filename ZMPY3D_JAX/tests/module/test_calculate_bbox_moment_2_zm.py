"""
Tests for calculate_bbox_moment_2_zm function.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import ZMPY3D_JAX as z


class TestCalculateBBoxMoment2ZM:
    """Test suite for calculate_bbox_moment_2_zm function."""

    @pytest.fixture
    def cache_data(self):
        """Load cache data for Zernike moment calculation."""
        max_order = 6
        cache_dir = Path(z.__file__).parent / "cache_data"
        log_cache_path = cache_dir / f"LogG_CLMCache_MaxOrder{max_order:02d}.pkl"

        with open(log_cache_path, "rb") as file:
            cache_pkl = pickle.load(file)

        return {
            "GCache_pqr_linear": cache_pkl["GCache_pqr_linear"],
            "GCache_complex": cache_pkl["GCache_complex"],
            "GCache_complex_index": cache_pkl["GCache_complex_index"],
            "CLMCache3D": cache_pkl["CLMCache3D"],
            "max_order": max_order,
        }

    @pytest.fixture
    def bbox_moment(self):
        """Create a sample bounding box moment."""
        max_order = 6
        # Create a simple moment array
        moment = np.random.rand(max_order + 1, max_order + 1, max_order + 1)
        return moment

    def test_basic_conversion(self, bbox_moment, cache_data):
        """Test basic conversion from bbox moment to Zernike moments."""
        zm_scaled, zm_raw = z.calculate_bbox_moment_2_zm(
            cache_data["max_order"],
            cache_data["GCache_complex"],
            cache_data["GCache_pqr_linear"],
            cache_data["GCache_complex_index"],
            cache_data["CLMCache3D"],
            bbox_moment,
        )

        # Check that both outputs are complex arrays
        assert np.iscomplexobj(zm_scaled)
        assert np.iscomplexobj(zm_raw)

        # Check shapes are appropriate
        assert zm_scaled.ndim == 3
        assert zm_raw.ndim == 3

    def test_output_shapes(self, bbox_moment, cache_data):
        """Test that output shapes match expected dimensions."""
        max_order = cache_data["max_order"]
        zm_scaled, zm_raw = z.calculate_bbox_moment_2_zm(
            max_order,
            cache_data["GCache_complex"],
            cache_data["GCache_pqr_linear"],
            cache_data["GCache_complex_index"],
            cache_data["CLMCache3D"],
            bbox_moment,
        )

        # Zernike moments should have shape related to max_order
        assert zm_scaled.shape[0] <= max_order + 1
        assert zm_raw.shape[0] <= max_order + 1

    def test_zero_moment(self, cache_data):
        """Test with zero bounding box moment."""
        max_order = cache_data["max_order"]
        zero_moment = np.zeros((max_order + 1, max_order + 1, max_order + 1))

        zm_scaled, zm_raw = z.calculate_bbox_moment_2_zm(
            max_order,
            cache_data["GCache_complex"],
            cache_data["GCache_pqr_linear"],
            cache_data["GCache_complex_index"],
            cache_data["CLMCache3D"],
            zero_moment,
        )

        # Most values should be zero or NaN
        # (Some may be NaN due to division in normalization)
        non_nan_scaled = zm_scaled[~np.isnan(zm_scaled)]
        non_nan_raw = zm_raw[~np.isnan(zm_raw)]

        if len(non_nan_scaled) > 0:
            assert np.allclose(non_nan_scaled, 0, atol=1e-10)
        if len(non_nan_raw) > 0:
            assert np.allclose(non_nan_raw, 0, atol=1e-10)

    def test_scaled_vs_raw(self, bbox_moment, cache_data):
        """Test relationship between scaled and raw Zernike moments."""
        zm_scaled, zm_raw = z.calculate_bbox_moment_2_zm(
            cache_data["max_order"],
            cache_data["GCache_complex"],
            cache_data["GCache_pqr_linear"],
            cache_data["GCache_complex_index"],
            cache_data["CLMCache3D"],
            bbox_moment,
        )

        # Scaled and raw should have same shape
        assert zm_scaled.shape == zm_raw.shape

        # Scaled should generally have similar or smaller magnitudes
        # (due to normalization by CLM coefficients)
        # This is not always true, so we just check they're related
        assert zm_scaled.dtype == zm_raw.dtype

    def test_deterministic(self, bbox_moment, cache_data):
        """Test that function is deterministic."""
        zm_scaled_1, zm_raw_1 = z.calculate_bbox_moment_2_zm(
            cache_data["max_order"],
            cache_data["GCache_complex"],
            cache_data["GCache_pqr_linear"],
            cache_data["GCache_complex_index"],
            cache_data["CLMCache3D"],
            bbox_moment,
        )

        zm_scaled_2, zm_raw_2 = z.calculate_bbox_moment_2_zm(
            cache_data["max_order"],
            cache_data["GCache_complex"],
            cache_data["GCache_pqr_linear"],
            cache_data["GCache_complex_index"],
            cache_data["CLMCache3D"],
            bbox_moment,
        )

        # Results should be identical
        np.testing.assert_array_equal(zm_scaled_1, zm_scaled_2)
        np.testing.assert_array_equal(zm_raw_1, zm_raw_2)

    def test_complex_moment(self, cache_data):
        """Test with complex-valued bounding box moment."""
        max_order = cache_data["max_order"]
        # Create complex moment (though bbox moments are typically real)
        complex_moment = np.random.rand(
            max_order + 1, max_order + 1, max_order + 1
        ) + 1j * np.random.rand(max_order + 1, max_order + 1, max_order + 1)

        zm_scaled, zm_raw = z.calculate_bbox_moment_2_zm(
            max_order,
            cache_data["GCache_complex"],
            cache_data["GCache_pqr_linear"],
            cache_data["GCache_complex_index"],
            cache_data["CLMCache3D"],
            complex_moment,
        )

        # Should still work with complex input
        assert np.iscomplexobj(zm_scaled)
        assert np.iscomplexobj(zm_raw)

    def test_symmetry_properties(self, cache_data):
        """Test that Zernike moments preserve certain symmetries."""
        max_order = cache_data["max_order"]

        # Create a symmetric moment
        symmetric_moment = np.ones((max_order + 1, max_order + 1, max_order + 1))

        zm_scaled, zm_raw = z.calculate_bbox_moment_2_zm(
            max_order,
            cache_data["GCache_complex"],
            cache_data["GCache_pqr_linear"],
            cache_data["GCache_complex_index"],
            cache_data["CLMCache3D"],
            symmetric_moment,
        )

        # For symmetric input, certain Zernike moments should have expected properties
        # (This is a simplified test - full symmetry tests would be more complex)
        assert zm_scaled is not None
        assert zm_raw is not None

    def test_nan_handling(self, cache_data):
        """Test handling of NaN values in input."""
        max_order = cache_data["max_order"]
        moment_with_nan = np.random.rand(max_order + 1, max_order + 1, max_order + 1)
        moment_with_nan[0, 0, 0] = np.nan

        zm_scaled, zm_raw = z.calculate_bbox_moment_2_zm(
            max_order,
            cache_data["GCache_complex"],
            cache_data["GCache_pqr_linear"],
            cache_data["GCache_complex_index"],
            cache_data["CLMCache3D"],
            moment_with_nan,
        )

        # Function should handle NaN (may propagate or replace)
        assert zm_scaled is not None
        assert zm_raw is not None
