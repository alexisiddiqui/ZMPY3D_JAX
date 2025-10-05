import chex
import jax.numpy as jnp
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

        # convert to JAX arrays where appropriate and assert shapes
        percentile_list = jnp.asarray(percentile_list)
        std_xyz = jnp.asarray(std_xyz)
        s = jnp.asarray(s)
        k = jnp.asarray(k)

        chex.assert_shape(percentile_list, (9, 1))
        chex.assert_shape(std_xyz, ())
        chex.assert_shape(s, ())
        chex.assert_shape(k, ())

    def test_symmetric_distribution(self, simple_coordinates):
        """Test with symmetric coordinates around origin."""
        percentile_list, std_xyz, s, k = get_ca_distance_info(simple_coordinates)

        percentile_list = jnp.asarray(percentile_list)
        std_xyz = jnp.asarray(std_xyz)
        s = jnp.asarray(s)

        # For symmetric distribution, skewness should be close to 0.
        # If all distances are identical std == 0, skewness may be NaN; allow that.
        if bool(jnp.isfinite(s)):
            # convert to Python float for magnitude check
            assert abs(float(s)) < 0.5, f"Skewness should be near 0 for symmetric data, got {s}"
        else:
            assert float(std_xyz) == 0.0, "Skewness is NaN only when std is zero"

        # All distances should be equal (unit sphere centered at origin)
        chex.assert_trees_all_close(percentile_list, jnp.ones_like(percentile_list), atol=1e-10)

    def test_percentile_ordering(self, random_coordinates):
        """Test that percentiles are in ascending order."""
        percentile_list, _, _, _ = get_ca_distance_info(random_coordinates)

        percentiles = jnp.ravel(jnp.asarray(percentile_list))
        ordering_ok = jnp.all(percentiles[:-1] <= percentiles[1:])
        assert bool(ordering_ok), "Percentiles should be in ascending order"

    def test_std_positive(self, random_coordinates):
        """Test that standard deviation is positive."""
        _, std_xyz, _, _ = get_ca_distance_info(random_coordinates)
        std_xyz = jnp.asarray(std_xyz)
        assert float(std_xyz) > 0.0, "Standard deviation should be positive"

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

        chex.assert_shape(jnp.asarray(percentile_list), (9, 1))
        # ensure scalars are finite / not NaN
        assert bool(jnp.isfinite(jnp.asarray(std_xyz)))
        assert bool(jnp.isfinite(jnp.asarray(s)))
        assert bool(jnp.isfinite(jnp.asarray(k)))

    def test_collinear_points(self, linear_coordinates):
        """Test with collinear points."""
        percentile_list, std_xyz, s, k = get_ca_distance_info(linear_coordinates)

        # All should be valid numbers (no NaN or inf)
        chex.assert_tree_all_finite({"p": percentile_list, "std": std_xyz, "s": s, "k": k})

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
        assert float(std_xyz) < 1e-9, "Standard deviation should be very small for clustered points"

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
        percentile_list = jnp.asarray(percentile_list)

        # Center should be at (3, 0, 0), distances should be [2, 1, 0, 1, 2]
        expected_median = 1.0  # 50th percentile
        median_idx = 4  # 50th percentile is at index 4 (0-indexed)

        chex.assert_trees_all_close(
            percentile_list[median_idx, 0],
            jnp.asarray(expected_median, dtype=percentile_list.dtype),
            atol=0.1,
        )

    def test_percentile_range(self, random_coordinates):
        """Test that percentiles are within the range of actual distances."""
        percentile_list, _, _, _ = get_ca_distance_info(random_coordinates)
        percentile_list = jnp.asarray(percentile_list).reshape(-1)

        # compute center and distances with jax.numpy for compatibility
        xyz_center = jnp.mean(jnp.asarray(random_coordinates), axis=0)
        xyz_dist2center = jnp.sqrt(
            jnp.sum((jnp.asarray(random_coordinates) - xyz_center) ** 2, axis=1)
        )

        min_dist = float(jnp.min(xyz_dist2center))
        max_dist = float(jnp.max(xyz_dist2center))

        assert bool(jnp.all(percentile_list >= min_dist)), "Percentiles should be >= min distance"
        assert bool(jnp.all(percentile_list <= max_dist)), "Percentiles should be <= max distance"

    def test_reproducibility(self, random_coordinates):
        """Test that function produces consistent results."""
        result1 = get_ca_distance_info(random_coordinates)
        result2 = get_ca_distance_info(random_coordinates)

        chex.assert_trees_all_close(result1, result2, atol=1e-12)
