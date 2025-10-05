"""
Tests for calculate_bbox_moment function.
"""

import sys
from pathlib import Path

import chex
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import ZMPY3D_JAX as z
from ZMPY3D_JAX.config import FLOAT_DTYPE

z.configure_for_scientific_computing()


class TestCalculateBBoxMoment:
    """Test suite for calculate_bbox_moment function."""

    @pytest.fixture
    def simple_voxel(self):
        """Create a simple 3D voxel grid for testing."""
        voxel = jnp.zeros((10, 10, 10), dtype=FLOAT_DTYPE)
        # Add some density in the center (use JAX immutable update)
        voxel = voxel.at[4:6, 4:6, 4:6].set(1.0)
        return voxel

    @pytest.fixture
    def xyz_samples(self):
        """Create sample coordinate arrays."""
        return {
            "X_sample": jnp.arange(11, dtype=FLOAT_DTYPE),
            "Y_sample": jnp.arange(11, dtype=FLOAT_DTYPE),
            "Z_sample": jnp.arange(11, dtype=FLOAT_DTYPE),
        }

    def test_order_1_moment(self, simple_voxel, xyz_samples):
        """Test calculation of first-order moments (mass, center)."""
        volume_mass, center, moment = z.calculate_bbox_moment(simple_voxel, 1, xyz_samples)

        # Volume mass should be positive
        assert float(volume_mass) > 0

        # Center should be 3D
        chex.assert_shape(center, (3,))

        # Center should be near the middle of the voxel
        assert bool(jnp.all(center >= 4.0))
        assert bool(jnp.all(center <= 6.0))

        # Moment array should have expected shape
        chex.assert_shape(moment, (2, 2, 2))

    def test_higher_order_moments(self, simple_voxel, xyz_samples):
        """Test calculation of higher-order moments."""
        max_order = 6
        _, _, moment = z.calculate_bbox_moment(simple_voxel, max_order, xyz_samples)

        # Moment array should have shape (max_order+1)^3
        expected_shape = (max_order + 1, max_order + 1, max_order + 1)
        chex.assert_shape(moment, expected_shape)

        # First element should be the total mass
        assert float(moment[0, 0, 0]) > 0

    def test_empty_voxel(self, xyz_samples):
        """Test with an empty voxel grid."""
        empty_voxel = jnp.zeros((10, 10, 10), dtype=FLOAT_DTYPE)
        volume_mass, center, moment = z.calculate_bbox_moment(empty_voxel, 1, xyz_samples)

        # Volume mass should be zero
        assert bool(jnp.isclose(volume_mass, 0.0))

        # Center should contain NaN or be zero
        assert bool(jnp.all(jnp.isnan(center))) or bool(jnp.all(center == 0))

    def test_uniform_voxel(self):
        """Test with a uniformly filled voxel."""
        uniform_voxel = jnp.ones((8, 8, 8), dtype=FLOAT_DTYPE)
        xyz_samples = {
            "X_sample": jnp.arange(9, dtype=FLOAT_DTYPE),
            "Y_sample": jnp.arange(9, dtype=FLOAT_DTYPE),
            "Z_sample": jnp.arange(9, dtype=FLOAT_DTYPE),
        }

        volume_mass, center, _ = z.calculate_bbox_moment(uniform_voxel, 1, xyz_samples)

        # Center should be at geometric center
        # For 8x8x8 voxel with coordinates 0-8, center is at coordinate 4.0
        expected_center = 4.0
        assert bool(jnp.allclose(center, expected_center, rtol=0.1))

    def test_different_max_orders(self, simple_voxel, xyz_samples):
        """Test with different maximum orders."""
        for max_order in [1, 3, 6, 10]:
            _, _, moment = z.calculate_bbox_moment(simple_voxel, max_order, xyz_samples)
            expected_shape = (max_order + 1, max_order + 1, max_order + 1)
            chex.assert_shape(moment, expected_shape)

    def test_asymmetric_voxel(self):
        """Test with an asymmetric voxel distribution."""
        voxel = jnp.zeros((10, 10, 10), dtype=FLOAT_DTYPE)
        # Place mass off-center using JAX immutable update
        voxel = voxel.at[7:9, 7:9, 7:9].set(2.0)

        xyz_samples = {
            "X_sample": jnp.arange(11, dtype=FLOAT_DTYPE),
            "Y_sample": jnp.arange(11, dtype=FLOAT_DTYPE),
            "Z_sample": jnp.arange(11, dtype=FLOAT_DTYPE),
        }

        _, center, _ = z.calculate_bbox_moment(voxel, 1, xyz_samples)

        # Center should be shifted toward the mass
        assert bool(jnp.all(center > 5.0))

    def test_output_types(self, simple_voxel, xyz_samples):
        """Test that output types are correct."""
        volume_mass, center, moment = z.calculate_bbox_moment(simple_voxel, 2, xyz_samples)

        assert isinstance(float(volume_mass), float)
        chex.assert_shape(center, (3,))
        chex.assert_shape(moment, (3, 3, 3))
        assert center.dtype == jnp.float64
        assert moment.dtype == jnp.float64
