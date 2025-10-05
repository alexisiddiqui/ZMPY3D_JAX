"""
Tests for calculate_bbox_moment function.
"""

import sys
from pathlib import Path
import os
import time
from datetime import datetime
import logging
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import ZMPY3D_JAX as z


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default


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

    def test_time_evaluation_runtime(self, simple_voxel, xyz_samples):
        """Timed evaluation wrapper for calculate_bbox_moment (order 1)."""
        repeats = _env_int("ZMPY3D_TIME_REPEATS", 100)
        max_seconds = _env_int("ZMPY3D_TIME_MAX_SEC", 1200)

        # Warm-up
        _ = z.calculate_bbox_moment(simple_voxel, 1, xyz_samples)

        start = time.perf_counter()
        for _ in range(repeats):
            _ = z.calculate_bbox_moment(simple_voxel, 1, xyz_samples)
        elapsed = time.perf_counter() - start

        # Save detailed timing information to log file (match test_time_simple.py style)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        benchmark_dir = os.path.join(script_dir, "_simple_time_benchmark")
        os.makedirs(benchmark_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"bbox_moment_benchmark_{timestamp}.log"
        log_filepath = os.path.join(benchmark_dir, log_filename)

        with open(log_filepath, "w") as log_file:
            log_file.write("ZMPY3D_JAX BBox Moment Benchmark\n")
            log_file.write("=" * 60 + "\n")
            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write("\nConfiguration:\n")
            log_file.write(f"  Repeats: {repeats}\n")
            log_file.write(f"  Max seconds: {max_seconds}\n")
            log_file.write("\nInput:\n")
            log_file.write(f"  Voxel shape: {simple_voxel.shape}\n")
            log_file.write(f"  XYZ sample sizes: {{'X': {len(xyz_samples['X_sample'])}, 'Y': {len(xyz_samples['Y_sample'])}, 'Z': {len(xyz_samples['Z_sample'])}}}\n")
            log_file.write("\nResults:\n")
            log_file.write(f"  Total time: {elapsed:.6f}s\n")
            log_file.write(f"  Time per iteration: {elapsed / repeats * 1000:.6f}ms\n")
            log_file.write(f"  Iterations per second: {repeats / elapsed:.2f}\n")
            log_file.write(f"\nStatus: {'PASS' if elapsed <= max_seconds else 'FAIL'}\n")
            if elapsed > max_seconds:
                log_file.write(f"  Exceeded threshold by: {elapsed - max_seconds:.2f}s\n")

        logging.info("BBox moment benchmark saved to: %s", log_filepath)
        assert elapsed <= max_seconds, f"Elapsed {elapsed:.2f}s exceeded {max_seconds}s"
