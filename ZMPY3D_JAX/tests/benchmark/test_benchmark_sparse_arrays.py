"""
Benchmark tests for sparse array support in ZMPY3D_JAX functions.
Tests the performance impact of using JAX sparse arrays vs dense arrays.
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import ZMPY3D_JAX as z


def _env_int(key: str, default: int) -> int:
    """Read integer from env with fallback."""
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default


class TestBBoxMomentWithSparseArrays:
    """Test suite for calculate_bbox_moment with sparse voxel arrays."""

    @pytest.fixture
    def dense_voxel(self):
        """Create a dense 3D voxel grid for testing."""
        voxel = np.zeros((10, 10, 10))
        # Add some density in the center (sparse pattern)
        voxel[4:6, 4:6, 4:6] = 1.0
        return jnp.asarray(voxel)

    @pytest.fixture
    def sparse_voxel(self, dense_voxel):
        """Convert dense voxel to sparse format using JAX sparse BCOO."""
        # BCOO (Batched COO) supports N-dimensional arrays
        return sparse.BCOO.fromdense(dense_voxel)

    @pytest.fixture
    def xyz_samples(self):
        """Create sample coordinate arrays."""
        return {
            "X_sample": jnp.arange(11, dtype=jnp.float32),
            "Y_sample": jnp.arange(11, dtype=jnp.float32),
            "Z_sample": jnp.arange(11, dtype=jnp.float32),
        }

    def test_sparse_vs_dense_functionality(self, dense_voxel, sparse_voxel, xyz_samples):
        """Test that sparse and dense inputs produce same results."""
        # Process dense voxel
        volume_dense, center_dense, moment_dense = z.calculate_bbox_moment(
            dense_voxel, 1, xyz_samples
        )

        # Process sparse voxel
        volume_sparse, center_sparse, moment_sparse = z.calculate_bbox_moment(
            sparse_voxel, 1, xyz_samples
        )

        # Results should be the same
        np.testing.assert_allclose(volume_dense, volume_sparse, rtol=1e-5)
        np.testing.assert_allclose(center_dense, center_sparse, rtol=1e-5)
        np.testing.assert_allclose(moment_dense, moment_sparse, rtol=1e-5)

    def test_sparse_array_benchmark(self, sparse_voxel, xyz_samples):
        """Benchmark calculate_bbox_moment with sparse input."""
        repeats = _env_int("ZMPY3D_TIME_REPEATS", 100)
        max_seconds = _env_int("ZMPY3D_TIME_MAX_SEC", 1200)

        # Warm-up
        _ = z.calculate_bbox_moment(sparse_voxel, 1, xyz_samples)

        start = time.perf_counter()
        for _ in range(repeats):
            _ = z.calculate_bbox_moment(sparse_voxel, 1, xyz_samples)
        elapsed = time.perf_counter() - start

        # Save timing
        script_dir = os.path.dirname(os.path.abspath(__file__))
        benchmark_dir = os.path.join(script_dir, "_simple_time_benchmark")
        os.makedirs(benchmark_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"bbox_moment_sparse_benchmark_{timestamp}.log"
        log_filepath = os.path.join(benchmark_dir, log_filename)

        with open(log_filepath, "w") as log_file:
            log_file.write("ZMPY3D_JAX BBox Moment Sparse Benchmark\n")
            log_file.write("=" * 60 + "\n")
            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write("\nConfiguration:\n")
            log_file.write(f"  Repeats: {repeats}\n")
            log_file.write(f"  Max seconds: {max_seconds}\n")
            log_file.write("\nInput:\n")
            log_file.write(f"  Voxel shape: {sparse_voxel.shape}\n")
            log_file.write(f"  Voxel format: BCOO sparse\n")
            log_file.write(f"  Non-zero elements: {sparse_voxel.nse}\n")
            log_file.write(f"  Sparsity: {100 * (1 - sparse_voxel.nse / np.prod(sparse_voxel.shape)):.1f}%\n")
            log_file.write("\nResults:\n")
            log_file.write(f"  Total time: {elapsed:.6f}s\n")
            log_file.write(f"  Time per iteration: {elapsed / repeats * 1000:.6f}ms\n")
            log_file.write(f"  Iterations per second: {repeats / elapsed:.2f}\n")
            log_file.write(f"\nStatus: {'PASS' if elapsed <= max_seconds else 'FAIL'}\n")
            if elapsed > max_seconds:
                log_file.write(f"  Exceeded threshold by: {elapsed - max_seconds:.2f}s\n")

        logging.info("BBox moment sparse benchmark saved to: %s", log_filepath)
        assert elapsed <= max_seconds, f"Elapsed {elapsed:.2f}s exceeded {max_seconds}s"


class TestMolecularRadiusWithSparseArrays:
    """Test suite for calculate_molecular_radius with sparse voxel arrays."""

    @pytest.fixture
    def dense_voxel(self):
        """Create a dense 3D voxel grid."""
        voxel = np.zeros((10, 10, 10))
        # Create a sparse pattern
        for i in range(10):
            for j in range(10):
                if (i + j) % 2 == 0:
                    voxel[i, j, 5] = 0.5 + np.random.random() * 0.5
        return jnp.asarray(voxel)

    @pytest.fixture
    def sparse_voxel(self, dense_voxel):
        """Convert dense voxel to sparse format using JAX sparse BCOO."""
        # BCOO (Batched COO) supports N-dimensional arrays
        return sparse.BCOO.fromdense(dense_voxel)

    @pytest.fixture
    def calculation_params(self, dense_voxel):
        """Calculate parameters needed for molecular radius calculation."""
        center = np.array([5.0, 5.0, 5.0])
        volume_mass = 10.0
        radius_multiplier = 1.0
        return center, volume_mass, radius_multiplier

    def test_sparse_vs_dense_functionality(
        self, dense_voxel, sparse_voxel, calculation_params
    ):
        """Test that sparse and dense inputs produce same results."""
        center, volume_mass, radius_multiplier = calculation_params

        # Process dense voxel
        avg_dense, max_dense = z.calculate_molecular_radius(
            dense_voxel, center, volume_mass, radius_multiplier
        )

        # Process sparse voxel
        avg_sparse, max_sparse = z.calculate_molecular_radius(
            sparse_voxel, center, volume_mass, radius_multiplier
        )

        # Results should be the same
        np.testing.assert_allclose(avg_dense, avg_sparse, rtol=1e-5)
        np.testing.assert_allclose(max_dense, max_sparse, rtol=1e-5)

    def test_sparse_array_benchmark(self, sparse_voxel, calculation_params):
        """Benchmark calculate_molecular_radius with sparse input."""
        center, volume_mass, radius_multiplier = calculation_params
        repeats = _env_int("ZMPY3D_TIME_REPEATS", 1000)
        max_seconds = _env_int("ZMPY3D_TIME_MAX_SEC", 1200)

        # Warm-up
        _ = z.calculate_molecular_radius(
            sparse_voxel, center, volume_mass, radius_multiplier
        )

        start = time.perf_counter()
        for _ in range(repeats):
            _ = z.calculate_molecular_radius(
                sparse_voxel, center, volume_mass, radius_multiplier
            )
        elapsed = time.perf_counter() - start

        # Save timing
        script_dir = os.path.dirname(os.path.abspath(__file__))
        benchmark_dir = os.path.join(script_dir, "_simple_time_benchmark")
        os.makedirs(benchmark_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"molecular_radius_sparse_benchmark_{timestamp}.log"
        log_filepath = os.path.join(benchmark_dir, log_filename)

        with open(log_filepath, "w") as log_file:
            log_file.write("ZMPY3D_JAX Molecular Radius Sparse Benchmark\n")
            log_file.write("=" * 60 + "\n")
            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write("\nConfiguration:\n")
            log_file.write(f"  Repeats: {repeats}\n")
            log_file.write(f"  Max seconds: {max_seconds}\n")
            log_file.write("\nInput:\n")
            log_file.write(f"  Voxel shape: {sparse_voxel.shape}\n")
            log_file.write(f"  Voxel format: BCOO sparse\n")
            log_file.write(f"  Non-zero elements: {sparse_voxel.nse}\n")
            log_file.write(f"  Sparsity: {100 * (1 - sparse_voxel.nse / np.prod(sparse_voxel.shape)):.1f}%\n")
            log_file.write("\nResults:\n")
            log_file.write(f"  Total time: {elapsed:.6f}s\n")
            log_file.write(f"  Time per iteration: {elapsed / repeats * 1000:.6f}ms\n")
            log_file.write(f"  Iterations per second: {repeats / elapsed:.2f}\n")
            log_file.write(f"\nStatus: {'PASS' if elapsed <= max_seconds else 'FAIL'}\n")
            if elapsed > max_seconds:
                log_file.write(f"  Exceeded threshold by: {elapsed - max_seconds:.2f}s\n")

        logging.info("Molecular radius sparse benchmark saved to: %s", log_filepath)
        assert elapsed <= max_seconds, f"Elapsed {elapsed:.2f}s exceeded {max_seconds}s"
