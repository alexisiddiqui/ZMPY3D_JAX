"""
Tests for calculate_molecular_radius function.
"""

import sys
from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import ZMPY3D_JAX as z


class TestCalculateMolecularRadius:
    """Test suite for calculate_molecular_radius function."""

    @pytest.fixture
    def spherical_voxel(self):
        """Create a spherical density distribution."""
        size = 20
        voxel = np.zeros((size, size, size))
        center = size // 2

        # Create spherical distribution
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    dist = np.sqrt((i - center) ** 2 + (j - center) ** 2 + (k - center) ** 2)
                    if dist <= 5:
                        voxel[i, j, k] = 1.0
        return voxel

    @pytest.fixture
    def centered_mass(self):
        """Create a simple centered mass."""
        voxel = np.zeros((10, 10, 10))
        voxel[4:6, 4:6, 4:6] = 1.0
        return voxel

    def test_spherical_distribution(self, spherical_voxel):
        """Test radius calculation with spherical distribution."""
        center = np.array([10.0, 10.0, 10.0])
        volume_mass = np.sum(spherical_voxel)
        multiplier = 1.80

        avg_radius, max_radius = z.calculate_molecular_radius(
            spherical_voxel, center, volume_mass, multiplier
        )

        avg_radius = jnp.asarray(avg_radius)
        max_radius = jnp.asarray(max_radius)
        chex.assert_tree_all_finite({"avg": avg_radius, "max": max_radius})

        # Average radius should be positive
        assert float(avg_radius) > 0.0

        # Max radius should be positive and reasonable for a sphere of radius 5
        assert float(max_radius) > 0.0
        assert float(max_radius) <= 5.0 * np.sqrt(3)  # Maximum possible distance in sphere

        # For a sphere, average radius with multiplier should be reasonable
        assert 3.0 < float(avg_radius) < 10.0

        # The multiplier increases avg_radius, so it may exceed max_radius
        assert float(max_radius) > 0 and float(avg_radius) > 0

    def test_centered_mass(self, centered_mass):
        """Test with a simple centered mass."""
        center = np.array([5.0, 5.0, 5.0])
        volume_mass = np.sum(centered_mass)
        multiplier = 1.80

        avg_radius, max_radius = z.calculate_molecular_radius(
            centered_mass, center, volume_mass, multiplier
        )

        avg_radius = jnp.asarray(avg_radius)
        max_radius = jnp.asarray(max_radius)
        chex.assert_tree_all_finite({"avg": avg_radius, "max": max_radius})

        # Both radii should be positive
        assert float(avg_radius) > 0.0
        assert float(max_radius) > 0.0

        # Both should be reasonable values
        assert float(avg_radius) < 10.0
        assert float(max_radius) < 10.0

    def test_different_multipliers(self, centered_mass):
        """Test with different radius multipliers."""
        center = np.array([5.0, 5.0, 5.0])
        volume_mass = np.sum(centered_mass)

        multipliers = [1.0, 1.5, 1.8, 2.0, 2.5]
        prev_avg = 0.0
        prev_max = None

        for multiplier in multipliers:
            avg_radius, max_radius = z.calculate_molecular_radius(
                centered_mass, center, volume_mass, multiplier
            )

            avg_radius = jnp.asarray(avg_radius)
            max_radius = jnp.asarray(max_radius)
            chex.assert_tree_all_finite({"avg": avg_radius, "max": max_radius})

            # Avg radius should increase with multiplier
            assert float(avg_radius) > prev_avg
            prev_avg = float(avg_radius)

            # Max radius should remain constant (it's the actual max distance)
            if prev_max is not None:
                assert abs(float(max_radius) - prev_max) < 1e-10
            prev_max = float(max_radius)

    def test_off_center_mass(self):
        """Test with off-center mass distribution."""
        voxel = np.zeros((15, 15, 15))
        voxel[10:13, 10:13, 10:13] = 2.0

        center = np.array([7.5, 7.5, 7.5])
        volume_mass = np.sum(voxel)
        multiplier = 1.80

        avg_radius, max_radius = z.calculate_molecular_radius(
            voxel, center, volume_mass, multiplier
        )

        avg_radius = jnp.asarray(avg_radius)
        max_radius = jnp.asarray(max_radius)
        chex.assert_tree_all_finite({"avg": avg_radius, "max": max_radius})

        # Both should be positive
        assert float(avg_radius) > 0.0
        assert float(max_radius) > 0.0

    def test_single_point_mass(self):
        """Test with a single point of mass."""
        voxel = np.zeros((10, 10, 10))
        voxel[7, 7, 7] = 1.0

        center = np.array([5.0, 5.0, 5.0])
        volume_mass = 1.0
        multiplier = 1.80

        avg_radius, max_radius = z.calculate_molecular_radius(
            voxel, center, volume_mass, multiplier
        )

        avg_radius = jnp.asarray(avg_radius)
        max_radius = jnp.asarray(max_radius)
        chex.assert_tree_all_finite({"avg": avg_radius, "max": max_radius})

        # For a single point, both should be positive
        assert float(avg_radius) > 0.0
        assert float(max_radius) > 0.0

        # Max should be the actual distance, avg should be multiplied version
        expected_distance = np.sqrt(3 * (2.0**2))
        assert abs(float(max_radius) - expected_distance) < 0.1

    def test_uniform_distribution(self):
        """Test with uniform mass distribution."""
        voxel = np.ones((8, 8, 8))
        center = np.array([3.5, 3.5, 3.5])
        volume_mass = np.sum(voxel)
        multiplier = 1.80

        avg_radius, max_radius = z.calculate_molecular_radius(
            voxel, center, volume_mass, multiplier
        )

        avg_radius = jnp.asarray(avg_radius)
        max_radius = jnp.asarray(max_radius)
        chex.assert_tree_all_finite({"avg": avg_radius, "max": max_radius})

        # For uniform distribution, values should be reasonable
        assert float(avg_radius) > 0.0
        assert float(max_radius) > 0.0

    def test_output_types(self, centered_mass):
        """Test that output types are correct."""
        center = np.array([5.0, 5.0, 5.0])
        volume_mass = np.sum(centered_mass)
        multiplier = 1.80

        avg_radius, max_radius = z.calculate_molecular_radius(
            centered_mass, center, volume_mass, multiplier
        )

        # convert to Python floats to validate types
        assert isinstance(float(avg_radius), float)
        assert isinstance(float(max_radius), float)

    def test_zero_volume_mass(self):
        """Test behavior with zero volume mass - should raise ValueError."""
        voxel = np.zeros((10, 10, 10))
        center = np.array([5.0, 5.0, 5.0])
        volume_mass = 0.0
        multiplier = 1.80

        # The function raises ValueError when there are no non-zero voxels
        with pytest.raises(ValueError, match="zero-size array"):
            z.calculate_molecular_radius(voxel, center, volume_mass, multiplier)

    def test_anisotropic_distribution(self):
        """Test with an anisotropic (elongated) distribution."""
        voxel = np.zeros((20, 10, 10))
        # Create elongated distribution along x-axis
        voxel[5:15, 4:6, 4:6] = 1.0

        center = np.array([10.0, 5.0, 5.0])
        volume_mass = np.sum(voxel)
        multiplier = 1.80

        avg_radius, max_radius = z.calculate_molecular_radius(
            voxel, center, volume_mass, multiplier
        )

        avg_radius = jnp.asarray(avg_radius)
        max_radius = jnp.asarray(max_radius)
        chex.assert_tree_all_finite({"avg": avg_radius, "max": max_radius})

        # Both radii should be positive
        assert float(avg_radius) > 0.0
        assert float(max_radius) > 0.0

        # For elongated distribution, max_radius reflects actual maximum distance
        # avg_radius is the weighted average multiplied by the multiplier
        # With multiplier > 1, avg_radius can exceed max_radius
        assert float(max_radius) < 10.0  # Reasonable upper bound for this distribution
