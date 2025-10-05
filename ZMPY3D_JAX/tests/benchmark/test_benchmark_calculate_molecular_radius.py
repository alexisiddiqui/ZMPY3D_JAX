"""
Tests for calculate_molecular_radius function.
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

        # Average radius should be positive
        assert avg_radius > 0

        # Max radius should be positive and reasonable for a sphere of radius 5
        assert max_radius > 0
        assert max_radius <= 5.0 * np.sqrt(3)  # Maximum possible distance in sphere

        # For a sphere, average radius with multiplier should be reasonable
        assert 3.0 < avg_radius < 10.0

        # The multiplier increases avg_radius, so it may exceed max_radius
        # Max radius is the actual maximum distance, avg is weighted and multiplied
        assert max_radius > 0 and avg_radius > 0

    def test_centered_mass(self, centered_mass):
        """Test with a simple centered mass."""
        center = np.array([5.0, 5.0, 5.0])
        volume_mass = np.sum(centered_mass)
        multiplier = 1.80

        avg_radius, max_radius = z.calculate_molecular_radius(
            centered_mass, center, volume_mass, multiplier
        )

        # Both radii should be positive
        assert avg_radius > 0
        assert max_radius > 0

        # Both should be reasonable values
        assert avg_radius < 10.0
        assert max_radius < 10.0

    def test_different_multipliers(self, centered_mass):
        """Test with different radius multipliers."""
        center = np.array([5.0, 5.0, 5.0])
        volume_mass = np.sum(centered_mass)

        multipliers = [1.0, 1.5, 1.8, 2.0, 2.5]
        prev_avg = 0
        prev_max = None

        for multiplier in multipliers:
            avg_radius, max_radius = z.calculate_molecular_radius(
                centered_mass, center, volume_mass, multiplier
            )

            # Avg radius should increase with multiplier
            assert avg_radius > prev_avg
            prev_avg = avg_radius

            # Max radius should remain constant (it's the actual max distance)
            if prev_max is not None:
                assert abs(max_radius - prev_max) < 1e-10
            prev_max = max_radius

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

        # Both should be positive
        assert avg_radius > 0
        assert max_radius > 0

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

        # For a single point, both should be positive
        assert avg_radius > 0
        assert max_radius > 0

        # Max should be the actual distance, avg should be multiplied version
        expected_distance = np.sqrt(3 * (2.0**2))
        assert abs(max_radius - expected_distance) < 0.1

    def test_uniform_distribution(self):
        """Test with uniform mass distribution."""
        voxel = np.ones((8, 8, 8))
        center = np.array([3.5, 3.5, 3.5])
        volume_mass = np.sum(voxel)
        multiplier = 1.80

        avg_radius, max_radius = z.calculate_molecular_radius(
            voxel, center, volume_mass, multiplier
        )

        # For uniform distribution, values should be reasonable
        assert avg_radius > 0
        assert max_radius > 0

    def test_output_types(self, centered_mass):
        """Test that output types are correct."""
        center = np.array([5.0, 5.0, 5.0])
        volume_mass = np.sum(centered_mass)
        multiplier = 1.80

        avg_radius, max_radius = z.calculate_molecular_radius(
            centered_mass, center, volume_mass, multiplier
        )

        assert isinstance(avg_radius, (float, np.floating))
        assert isinstance(max_radius, (float, np.floating))

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

        # Both radii should be positive
        assert avg_radius > 0
        assert max_radius > 0

        # For elongated distribution, max_radius reflects actual maximum distance
        # avg_radius is the weighted average multiplied by the multiplier
        # With multiplier > 1, avg_radius can exceed max_radius
        assert max_radius < 10.0  # Reasonable upper bound for this distribution

    def test_time_evaluation_runtime(self, centered_mass):
        """Timed evaluation wrapper for calculate_molecular_radius."""
        repeats = _env_int("ZMPY3D_TIME_REPEATS", 100)
        max_seconds = _env_int("ZMPY3D_TIME_MAX_SEC", 1200)

        center = np.array([5.0, 5.0, 5.0])
        volume_mass = np.sum(centered_mass)
        multiplier = 1.80

        # Warm-up
        _ = z.calculate_molecular_radius(centered_mass, center, volume_mass, multiplier)

        start = time.perf_counter()
        for _ in range(repeats):
            _ = z.calculate_molecular_radius(centered_mass, center, volume_mass, multiplier)
        elapsed = time.perf_counter() - start

        # Save detailed timing information to log file (like test_time_simple.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        benchmark_dir = os.path.join(script_dir, "_simple_time_benchmark")
        os.makedirs(benchmark_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"molecular_radius_benchmark_{timestamp}.log"
        log_filepath = os.path.join(benchmark_dir, log_filename)

        with open(log_filepath, "w") as log_file:
            log_file.write("ZMPY3D_JAX Molecular Radius Benchmark\n")
            log_file.write("=" * 60 + "\n")
            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write("\nConfiguration:\n")
            log_file.write(f"  Repeats: {repeats}\n")
            log_file.write(f"  Max seconds: {max_seconds}\n")
            log_file.write("\nInput:\n")
            log_file.write(f"  Voxel shape: {centered_mass.shape}\n")
            log_file.write("\nResults:\n")
            log_file.write(f"  Total time: {elapsed:.6f}s\n")
            log_file.write(f"  Time per iteration: {elapsed / repeats * 1000:.6f}ms\n")
            log_file.write(f"  Iterations per second: {repeats / elapsed:.2f}\n")
            log_file.write(f"\nStatus: {'PASS' if elapsed <= max_seconds else 'FAIL'}\n")
            if elapsed > max_seconds:
                log_file.write(f"  Exceeded threshold by: {elapsed - max_seconds:.2f}s\n")

        logging.info("Molecular radius benchmark saved to: %s", log_filepath)
        assert elapsed <= max_seconds, f"Elapsed {elapsed:.2f}s exceeded {max_seconds}s"
