import numpy as np
import pytest

from ZMPY3D_JAX.lib.get_ca_distance_info import get_ca_distance_info


class TestGetCADistanceInfo:
    """Test suite for get_ca_distance_info function."""

    @pytest.fixture
    def simple_coordinates(self):
        """Simple test coordinates - points on a unit sphere."""
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )

    @pytest.fixture
    def linear_coordinates(self):
        """Collinear points along x-axis."""
        return np.array([[i, 0.0, 0.0] for i in range(10)])

    @pytest.fixture
    def random_coordinates(self):
        """Random 3D coordinates for general testing."""
        np.random.seed(42)
        return np.random.randn(20, 3)

    def test_output_shapes(self, simple_coordinates):
        """Test that output shapes are correct."""
        percentile_list, std_xyz, s, k = get_ca_distance_info(simple_coordinates)

        assert percentile_list.shape == (9, 1), "Percentile list should be (9, 1)"
        assert isinstance(std_xyz, (float, np.floating)), "std should be a float"
        assert isinstance(s, (float, np.floating)), "skewness should be a float"
        assert isinstance(k, (float, np.floating)), "kurtosis should be a float"

    def test_symmetric_distribution(self, simple_coordinates):
        """Test with symmetric coordinates around origin."""
        percentile_list, std_xyz, s, k = get_ca_distance_info(simple_coordinates)

        # For symmetric distribution, skewness should be close to 0.
        # If all distances are identical std == 0, skewness may be NaN; allow that.
        if np.isfinite(s):
            assert abs(s) < 0.5, f"Skewness should be near 0 for symmetric data, got {s}"
        else:
            assert std_xyz == 0.0, "Skewness is NaN only when std is zero"

        # All distances should be equal (unit sphere centered at origin)
        assert np.allclose(percentile_list, 1.0, atol=1e-10), (
            "All percentiles should be ~1.0 for unit sphere"
        )

    def test_percentile_ordering(self, random_coordinates):
        """Test that percentiles are in ascending order."""
        percentile_list, _, _, _ = get_ca_distance_info(random_coordinates)

        percentiles = percentile_list.flatten()
        assert np.all(percentiles[:-1] <= percentiles[1:]), (
            "Percentiles should be in ascending order"
        )

    def test_std_positive(self, random_coordinates):
        """Test that standard deviation is positive."""
        _, std_xyz, _, _ = get_ca_distance_info(random_coordinates)

        assert std_xyz > 0, "Standard deviation should be positive"

    def test_minimum_points(self):
        """Test with minimum number of points (4 for kurtosis calculation)."""
        xyz = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        percentile_list, std_xyz, s, k = get_ca_distance_info(xyz)

        assert percentile_list.shape == (9, 1)
        assert not np.isnan(std_xyz)
        assert not np.isnan(s)
        assert not np.isnan(k)

    def test_collinear_points(self, linear_coordinates):
        """Test with collinear points."""
        percentile_list, std_xyz, s, k = get_ca_distance_info(linear_coordinates)

        # All should be valid numbers (no NaN or inf)
        assert np.all(np.isfinite(percentile_list))
        assert np.isfinite(std_xyz)
        assert np.isfinite(s)
        assert np.isfinite(k)

    def test_single_point_cluster(self):
        """Test with points very close together (near zero variance)."""
        xyz = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0 + 1e-10, 1.0, 1.0],
                [1.0, 1.0 + 1e-10, 1.0],
                [1.0, 1.0, 1.0 + 1e-10],
            ]
        )

        percentile_list, std_xyz, s, k = get_ca_distance_info(xyz)

        # Standard deviation should be very small
        assert std_xyz < 1e-9, "Standard deviation should be very small for clustered points"

    def test_known_statistical_values(self):
        """Test with data that has known statistical properties."""
        # Create points at specific distances from origin
        xyz = np.array(
            [
                [1.0, 0.0, 0.0],  # distance = 1
                [2.0, 0.0, 0.0],  # distance = 2
                [3.0, 0.0, 0.0],  # distance = 3
                [4.0, 0.0, 0.0],  # distance = 4
                [5.0, 0.0, 0.0],  # distance = 5
            ]
        )

        percentile_list, std_xyz, s, k = get_ca_distance_info(xyz)

        # Center should be at (3, 0, 0), distances should be [2, 1, 0, 1, 2]
        expected_median = 1.0  # 50th percentile
        median_idx = 4  # 50th percentile is at index 4 (0-indexed)

        assert np.isclose(percentile_list[median_idx, 0], expected_median, atol=0.1), (
            f"Median distance should be ~{expected_median}"
        )

    def test_percentile_range(self, random_coordinates):
        """Test that percentiles are within the range of actual distances."""
        percentile_list, _, _, _ = get_ca_distance_info(random_coordinates)

        xyz_center = np.mean(random_coordinates, axis=0)
        xyz_dist2center = np.sqrt(np.sum((random_coordinates - xyz_center) ** 2, axis=1))

        min_dist = np.min(xyz_dist2center)
        max_dist = np.max(xyz_dist2center)

        assert np.all(percentile_list >= min_dist), "Percentiles should be >= min distance"
        assert np.all(percentile_list <= max_dist), "Percentiles should be <= max distance"

    def test_reproducibility(self, random_coordinates):
        """Test that function produces consistent results."""
        result1 = get_ca_distance_info(random_coordinates)
        result2 = get_ca_distance_info(random_coordinates)

        np.testing.assert_array_almost_equal(result1[0], result2[0])
        assert result1[1] == result2[1]
        assert result1[2] == result2[2]
        assert result1[3] == result2[3]
