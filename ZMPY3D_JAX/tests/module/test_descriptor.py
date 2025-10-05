import numpy as np

from ZMPY3D_JAX.lib.get_3dzd_121_descriptor02 import get_3dzd_121_descriptor02


class TestGet3DZD121Descriptor02:
    """Test suite for get_3dzd_121_descriptor02 function."""

    def test_basic_functionality(self):
        """Test basic functionality with simple input."""
        # Create a simple 3D array (n, l, m) where n=2, l=2, m=3
        z_moment_scaled = np.array(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        )

        result = get_3dzd_121_descriptor02(z_moment_scaled)

        # Check output shape matches (n, l)
        assert result.shape == (2, 2)
        # Check all values are non-negative or NaN
        assert np.all((result >= 0) | np.isnan(result))

    def test_nan_handling(self):
        """Test that NaN values in input are converted to zeros."""
        z_moment_scaled = np.array(
            [[[np.nan, 2.0, 3.0], [4.0, np.nan, 6.0]], [[7.0, 8.0, np.nan], [10.0, 11.0, 12.0]]]
        )

        result = get_3dzd_121_descriptor02(z_moment_scaled)

        # Result should be valid (no errors from NaN in computation)
        assert result.shape == (2, 2)

    def test_zero_input(self):
        """Test with all zeros - should produce NaN output due to threshold."""
        z_moment_scaled = np.zeros((3, 3, 5))

        result = get_3dzd_121_descriptor02(z_moment_scaled)

        # All values should be NaN (below threshold)
        assert np.all(np.isnan(result))

    def test_small_values_threshold(self):
        """Test that very small values below 1e-20 are converted to NaN."""
        z_moment_scaled = np.array(
            [
                [[1e-15, 1e-15, 1e-15], [1e-15, 1e-15, 1e-15]],
                [[1e-15, 1e-15, 1e-15], [1e-15, 1e-15, 1e-15]],
            ]
        )

        result = get_3dzd_121_descriptor02(z_moment_scaled)

        # Each |1e-15|^2 = 1e-30
        # positive: 3*1e-30 = 3e-30, negative: 2*1e-30 = 2e-30
        # sqrt(5e-30) ≈ 2.236e-15, still above 1e-20
        # Need even smaller values: sqrt(positive + negative) < 1e-20
        # So positive + negative < 1e-40
        # With 3 values: need each value^2 * 5 < 1e-40
        # value^2 < 2e-41, value < ~1.4e-21
        # Using 1e-22 should work
        z_moment_scaled = np.array(
            [
                [[1e-22, 1e-22, 1e-22], [1e-22, 1e-22, 1e-22]],
                [[1e-22, 1e-22, 1e-22], [1e-22, 1e-22, 1e-22]],
            ]
        )

        result = get_3dzd_121_descriptor02(z_moment_scaled)

        # Values should be NaN (below 1e-20 threshold)
        assert np.all(np.isnan(result))

    def test_rotation_invariance_calculation(self):
        """Test the rotation invariance calculation logic."""
        # Simple case where we can verify calculation manually
        z_moment_scaled = np.array(
            [
                [[3.0, 4.0, 0.0]]  # |3|^2 + |4|^2 + |0|^2 = 9 + 16 + 0 = 25
            ]
        )

        result = get_3dzd_121_descriptor02(z_moment_scaled)

        # positive norm: 9 + 16 + 0 = 25
        # negative norm (first col zeroed): 0 + 16 + 0 = 16
        # total: sqrt(25 + 16) = sqrt(41) ≈ 6.403
        expected = np.sqrt(41)
        np.testing.assert_allclose(result[0, 0], expected, rtol=1e-10)

    def test_first_column_zeroing(self):
        """Test that the first column (m=0) is zeroed in negative norm calculation."""
        z_moment_scaled = np.array([[[10.0, 1.0, 1.0]]])

        result = get_3dzd_121_descriptor02(z_moment_scaled)

        # positive: 100 + 1 + 1 = 102
        # negative (first col zeroed): 0 + 1 + 1 = 2
        # total: sqrt(102 + 2) = sqrt(104)
        expected = np.sqrt(104)
        np.testing.assert_allclose(result[0, 0], expected, rtol=1e-10)

    def test_complex_values(self):
        """Test with complex input values."""
        z_moment_scaled = np.array([[[3.0 + 4.0j, 1.0, 2.0]]])

        result = get_3dzd_121_descriptor02(z_moment_scaled)

        # |3+4j|^2 = 25, |1|^2 = 1, |2|^2 = 4
        # positive: 25 + 1 + 4 = 30
        # negative: 0 + 1 + 4 = 5
        expected = np.sqrt(30 + 5)
        np.testing.assert_allclose(result[0, 0], expected, rtol=1e-10)

    def test_output_shape_preservation(self):
        """Test that output shape is (n, l) from input (n, l, m)."""
        for n, l, m in [(2, 3, 4), (5, 5, 7), (10, 8, 9)]:
            z_moment_scaled = np.random.randn(n, l, m) + 1j * np.random.randn(n, l, m)
            result = get_3dzd_121_descriptor02(z_moment_scaled)
            assert result.shape == (n, l)

    def test_input_not_modified(self):
        """Test that the function doesn't modify the input array."""
        z_moment_scaled = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=float)
        original = z_moment_scaled.copy()

        _ = get_3dzd_121_descriptor02(z_moment_scaled)

        # Input should remain unchanged
        np.testing.assert_array_equal(z_moment_scaled, original)
