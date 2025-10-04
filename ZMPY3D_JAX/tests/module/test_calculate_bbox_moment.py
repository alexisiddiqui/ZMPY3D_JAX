"""
Tests for calculate_bbox_moment function.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import ZMPY3D_JAX as z


class TestCalculateBBoxMoment:
    """Test suite for calculate_bbox_moment function."""

    @pytest.fixture
    def simple_voxel(self):
        """Create a simple 3D voxel grid for testing."""
        voxel = np.zeros((10, 10, 10))
        # Add some density in the center
        voxel[4:6, 4:6, 4:6] = 1.0
        return voxel

    @pytest.fixture
    def xyz_samples(self):
        """Create sample coordinate arrays."""
        return {
            "X_sample": np.arange(11, dtype=np.float64),
            "Y_sample": np.arange(11, dtype=np.float64),
            "Z_sample": np.arange(11, dtype=np.float64),
        }

    def test_order_1_moment(self, simple_voxel, xyz_samples):
        """Test calculation of first-order moments (mass, center)."""
        volume_mass, center, moment = z.calculate_bbox_moment(simple_voxel, 1, xyz_samples)

        # Volume mass should be positive
        assert volume_mass > 0

        # Center should be 3D
        assert center.shape == (3,)

        # Center should be near the middle of the voxel
        assert np.all(center >= 4.0)
        assert np.all(center <= 6.0)

        # Moment array should have expected shape
        assert moment.shape == (2, 2, 2)

    def test_higher_order_moments(self, simple_voxel, xyz_samples):
        """Test calculation of higher-order moments."""
        max_order = 6
        _, _, moment = z.calculate_bbox_moment(simple_voxel, max_order, xyz_samples)

        # Moment array should have shape (max_order+1)^3
        expected_shape = (max_order + 1, max_order + 1, max_order + 1)
        assert moment.shape == expected_shape

        # First element should be the total mass
        assert moment[0, 0, 0] > 0

    def test_empty_voxel(self, xyz_samples):
        """Test with an empty voxel grid."""
        empty_voxel = np.zeros((10, 10, 10))
        volume_mass, center, moment = z.calculate_bbox_moment(empty_voxel, 1, xyz_samples)

        # Volume mass should be zero
        assert volume_mass == 0

        # Center should contain NaN or be zero
        assert np.all(np.isnan(center)) or np.all(center == 0)

    def test_uniform_voxel(self):
        """Test with a uniformly filled voxel."""
        uniform_voxel = np.ones((8, 8, 8))
        xyz_samples = {
            "X_sample": np.arange(9, dtype=np.float64),
            "Y_sample": np.arange(9, dtype=np.float64),
            "Z_sample": np.arange(9, dtype=np.float64),
        }

        volume_mass, center, _ = z.calculate_bbox_moment(uniform_voxel, 1, xyz_samples)

        # Center should be at geometric center
        # For 8x8x8 voxel with coordinates 0-8, center is at coordinate 4.0
        expected_center = 4.0
        assert np.allclose(center, expected_center, rtol=0.1)

    def test_different_max_orders(self, simple_voxel, xyz_samples):
        """Test with different maximum orders."""
        for max_order in [1, 3, 6, 10]:
            _, _, moment = z.calculate_bbox_moment(simple_voxel, max_order, xyz_samples)
            expected_shape = (max_order + 1, max_order + 1, max_order + 1)
            assert moment.shape == expected_shape

    def test_asymmetric_voxel(self):
        """Test with an asymmetric voxel distribution."""
        voxel = np.zeros((10, 10, 10))
        # Place mass off-center
        voxel[7:9, 7:9, 7:9] = 2.0

        xyz_samples = {
            "X_sample": np.arange(11, dtype=np.float64),
            "Y_sample": np.arange(11, dtype=np.float64),
            "Z_sample": np.arange(11, dtype=np.float64),
        }

        _, center, _ = z.calculate_bbox_moment(voxel, 1, xyz_samples)

        # Center should be shifted toward the mass
        assert np.all(center > 5.0)

    def test_output_types(self, simple_voxel, xyz_samples):
        """Test that output types are correct."""
        volume_mass, center, moment = z.calculate_bbox_moment(simple_voxel, 2, xyz_samples)

        assert isinstance(volume_mass, (float, np.floating))
        assert isinstance(center, np.ndarray)
        assert isinstance(moment, np.ndarray)
        assert center.dtype == np.float64
        assert moment.dtype == np.float64
