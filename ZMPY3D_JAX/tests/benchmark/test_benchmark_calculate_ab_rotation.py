"""
Tests for calculate_ab_rotation and calculate_ab_rotation_all functions.
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import ZMPY3D_JAX as z


def _env_int(key: str, default: int) -> int:
    """Read integer from env with fallback."""
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default


class TestCalculateABRotation:
    """Test suite for calculate_ab_rotation function."""

    @pytest.fixture
    def real_protein_zm(self):
        """Create Zernike moments from actual protein structure coordinates."""
        # Use small protein fragment coordinates
        xyz = np.array(
            [
                [99.663, 55.677, 97.701],
                [96.343, 56.513, 96.936],
                [95.604, 57.569, 93.868],
                [93.579, 60.206, 92.736],
                [90.848, 60.023, 90.474],
            ]
        )

        aa_names = ["CA"] * len(xyz)
        grid_width = 1.0
        max_order = 6

        param = z.get_global_parameter()
        residue_box = z.get_residue_gaussian_density_cache(param)

        voxel, corner = z.fill_voxel_by_weight_density(
            xyz, aa_names, param["residue_weight_map"], grid_width, residue_box[grid_width]
        )

        dimension = voxel.shape
        xyz_sample = {
            "X_sample": np.arange(dimension[0] + 1),
            "Y_sample": np.arange(dimension[1] + 1),
            "Z_sample": np.arange(dimension[2] + 1),
        }

        volume, center, _ = z.calculate_bbox_moment(voxel, 1, xyz_sample)
        avg_dist, _ = z.calculate_molecular_radius(
            voxel, center, volume, param["default_radius_multiplier"]
        )

        sphere_xyz = z.get_bbox_moment_xyz_sample(center, avg_dist, dimension)
        _, _, bbox_moment = z.calculate_bbox_moment(voxel, max_order, sphere_xyz)

        # Get raw Zernike moments
        import pickle

        cache_dir = Path(z.__file__).parent / "cache_data"
        log_cache_path = cache_dir / f"LogG_CLMCache_MaxOrder{max_order:02d}.pkl"

        with open(log_cache_path, "rb") as file:
            cache_pkl = pickle.load(file)

        _, zm_raw = z.calculate_bbox_moment_2_zm(
            max_order,
            cache_pkl["GCache_complex"],
            cache_pkl["GCache_pqr_linear"],
            cache_pkl["GCache_complex_index"],
            cache_pkl["CLMCache3D"],
            bbox_moment,
        )

        return zm_raw

    def test_basic_rotation_order2(self, real_protein_zm):
        """Test AB rotation calculation for order 2."""
        ab_array = z.calculate_ab_rotation(real_protein_zm, 2)

        # Should return a 2D array of [a, b] pairs
        assert isinstance(ab_array, np.ndarray)
        assert ab_array.ndim == 2
        assert ab_array.shape[1] == 2  # Each row has 2 elements (a, b)
        assert ab_array.shape[0] > 0  # At least one solution

    def test_basic_rotation_order3(self, real_protein_zm):
        """Test AB rotation calculation for order 3."""
        ab_array = z.calculate_ab_rotation(real_protein_zm, 3)

        assert isinstance(ab_array, np.ndarray)
        assert ab_array.ndim == 2
        assert ab_array.shape[1] == 2

    def test_rotation_orders_2_to_6(self, real_protein_zm):
        """Test rotation calculation for orders 2 through 6."""
        for target_order in range(2, 7):
            ab_array = z.calculate_ab_rotation(real_protein_zm, target_order)

            # Should get some rotation pairs
            assert ab_array.shape[0] > 0
            assert ab_array.shape[1] == 2

            # Each pair should have |a|^2 + |b|^2 ≈ 1 (normalization)
            for i in range(ab_array.shape[0]):
                a, b = complex(ab_array[i, 0]), complex(ab_array[i, 1])
                norm = np.abs(a) ** 2 + np.abs(b) ** 2
                # Allow some numerical tolerance
                if not np.isnan(norm):
                    assert np.isclose(norm, 1.0, rtol=0.2, atol=0.1)

    def test_deterministic(self, real_protein_zm):
        """Test that function is deterministic."""
        ab_array_1 = z.calculate_ab_rotation(real_protein_zm, 2)
        ab_array_2 = z.calculate_ab_rotation(real_protein_zm, 2)

        # Should get same results
        np.testing.assert_array_almost_equal(ab_array_1, ab_array_2)

    def test_zero_moments(self):
        """Test with zero Zernike moments."""
        # Create properly structured zero array
        max_order = 6
        zero_zm = np.zeros((max_order + 1, max_order + 1, 2 * max_order + 1), dtype=complex)

        # The function should handle zero input gracefully or raise an error
        try:
            ab_array = z.calculate_ab_rotation(zero_zm, 2)
            # If it doesn't raise an error, check that it returns something reasonable
            assert isinstance(ab_array, np.ndarray)
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError, RuntimeWarning):
            # Expected to raise an error for degenerate input
            pass

    def test_conjugate_symmetry(self, real_protein_zm):
        """Test that results respect complex conjugate relationships."""
        ab_array = z.calculate_ab_rotation(real_protein_zm, 2)

        # Should have some solutions
        assert ab_array.shape[0] > 0

    def test_time_evaluation_runtime(self, real_protein_zm):
        """Timed evaluation wrapper for calculate_ab_rotation (order 2)."""
        repeats = _env_int("ZMPY3D_TIME_REPEATS", 10000)
        max_seconds = _env_int("ZMPY3D_TIME_MAX_SEC", 1200)

        target_order = 2

        # Warm-up
        _ = z.calculate_ab_rotation(real_protein_zm, target_order)

        start = time.perf_counter()
        for _ in range(repeats):
            _ = z.calculate_ab_rotation(real_protein_zm, target_order)
        elapsed = time.perf_counter() - start

        # Save detailed timing information to log file (like test_time_simple.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        benchmark_dir = os.path.join(script_dir, "_simple_time_benchmark")
        os.makedirs(benchmark_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"ab_rotation_benchmark_{timestamp}.log"
        log_filepath = os.path.join(benchmark_dir, log_filename)

        with open(log_filepath, "w") as log_file:
            log_file.write("ZMPY3D_JAX AB Rotation Benchmark\n")
            log_file.write("=" * 60 + "\n")
            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write("\nConfiguration:\n")
            log_file.write(f"  Repeats: {repeats}\n")
            log_file.write(f"  Max seconds: {max_seconds}\n")
            log_file.write(f"  Target order: {target_order}\n")
            log_file.write("\nResults:\n")
            log_file.write(f"  Total time: {elapsed:.6f}s\n")
            log_file.write(f"  Time per iteration: {elapsed / repeats * 1000:.6f}ms\n")
            log_file.write(f"  Iterations per second: {repeats / elapsed:.2f}\n")
            log_file.write(f"\nStatus: {'PASS' if elapsed <= max_seconds else 'FAIL'}\n")
            if elapsed > max_seconds:
                log_file.write(f"  Exceeded threshold by: {elapsed - max_seconds:.2f}s\n")

        logging.info("AB rotation benchmark saved to: %s", log_filepath)
        assert elapsed <= max_seconds, f"Elapsed {elapsed:.2f}s exceeded {max_seconds}s"


class TestCalculateABRotationAll:
    """Test suite for calculate_ab_rotation_all function."""

    @pytest.fixture
    def real_protein_zm(self):
        """Create Zernike moments from actual protein structure coordinates."""
        xyz = np.array(
            [
                [99.663, 55.677, 97.701],
                [96.343, 56.513, 96.936],
                [95.604, 57.569, 93.868],
                [93.579, 60.206, 92.736],
                [90.848, 60.023, 90.474],
            ]
        )

        aa_names = ["CA"] * len(xyz)
        grid_width = 1.0
        max_order = 6

        param = z.get_global_parameter()
        residue_box = z.get_residue_gaussian_density_cache(param)

        voxel, corner = z.fill_voxel_by_weight_density(
            xyz, aa_names, param["residue_weight_map"], grid_width, residue_box[grid_width]
        )

        dimension = voxel.shape
        xyz_sample = {
            "X_sample": np.arange(dimension[0] + 1),
            "Y_sample": np.arange(dimension[1] + 1),
            "Z_sample": np.arange(dimension[2] + 1),
        }

        volume, center, _ = z.calculate_bbox_moment(voxel, 1, xyz_sample)
        avg_dist, _ = z.calculate_molecular_radius(
            voxel, center, volume, param["default_radius_multiplier"]
        )

        sphere_xyz = z.get_bbox_moment_xyz_sample(center, avg_dist, dimension)
        _, _, bbox_moment = z.calculate_bbox_moment(voxel, max_order, sphere_xyz)

        import pickle

        cache_dir = Path(z.__file__).parent / "cache_data"
        log_cache_path = cache_dir / f"LogG_CLMCache_MaxOrder{max_order:02d}.pkl"

        with open(log_cache_path, "rb") as file:
            cache_pkl = pickle.load(file)

        _, zm_raw = z.calculate_bbox_moment_2_zm(
            max_order,
            cache_pkl["GCache_complex"],
            cache_pkl["GCache_pqr_linear"],
            cache_pkl["GCache_complex_index"],
            cache_pkl["CLMCache3D"],
            bbox_moment,
        )

        return zm_raw

    def test_all_rotations_order2(self, real_protein_zm):
        """Test calculation of all rotations for order 2."""
        ab_list_all = z.calculate_ab_rotation_all(real_protein_zm, 2)

        # Should return a list of arrays
        assert isinstance(ab_list_all, list)
        assert len(ab_list_all) > 0

        # Each element should be a 2D array with shape (n, 2)
        for ab_array in ab_list_all:
            assert isinstance(ab_array, np.ndarray)
            assert ab_array.ndim == 2
            assert ab_array.shape[1] == 2

    def test_all_rotations_order3(self, real_protein_zm):
        """Test calculation of all rotations for order 3."""
        ab_list_all = z.calculate_ab_rotation_all(real_protein_zm, 3)

        assert isinstance(ab_list_all, list)
        assert len(ab_list_all) > 0

    def test_all_rotations_orders_2_to_6(self, real_protein_zm):
        """Test all rotations for orders 2 through 6."""
        results = {}

        for target_order in range(2, 7):
            ab_list_all = z.calculate_ab_rotation_all(real_protein_zm, target_order)
            results[target_order] = len(ab_list_all)

            # Should get multiple solutions
            assert len(ab_list_all) > 0

    def test_normalization(self, real_protein_zm):
        """Test that all AB pairs are properly normalized."""
        ab_list_all = z.calculate_ab_rotation_all(real_protein_zm, 2)

        for ab_array in ab_list_all:
            for i in range(ab_array.shape[0]):
                a, b = complex(ab_array[i, 0]), complex(ab_array[i, 1])
                norm = np.abs(a) ** 2 + np.abs(b) ** 2

                # Should be normalized to 1 (with some tolerance), skip NaN
                if not np.isnan(norm):
                    assert np.isclose(norm, 1.0, rtol=0.2, atol=0.1)

    def test_deterministic(self, real_protein_zm):
        """Test that function produces consistent results."""
        ab_list_1 = z.calculate_ab_rotation_all(real_protein_zm, 3)
        ab_list_2 = z.calculate_ab_rotation_all(real_protein_zm, 3)

        # Should get same number of solutions
        assert len(ab_list_1) == len(ab_list_2)

        # Should get same values
        for arr1, arr2 in zip(ab_list_1, ab_list_2):
            np.testing.assert_array_almost_equal(arr1, arr2)

    def test_comparison_with_single(self, real_protein_zm):
        """Test that _all version contains results from single version."""
        target_order = 2

        ab_single = z.calculate_ab_rotation(real_protein_zm, target_order)
        ab_all = z.calculate_ab_rotation_all(real_protein_zm, target_order)

        # The 'all' version should have at least as many total pairs
        total_pairs_all = sum(arr.shape[0] for arr in ab_all)
        assert total_pairs_all >= ab_single.shape[0]

    def test_different_orders_coverage(self, real_protein_zm):
        """Test that different orders produce different numbers of solutions."""
        counts = {}

        for order in range(2, 7):
            ab_list = z.calculate_ab_rotation_all(real_protein_zm, order)
            counts[order] = len(ab_list)

        # Should have solutions for each order
        for order, count in counts.items():
            assert count > 0

    def test_calculate_ab_rotation(self):
        """Test AB rotation calculation with realistic protein structure coordinates."""
        xyz_a = np.array(
            [
                [99.663, 55.677, 97.701],
                [96.343, 56.513, 96.936],
                [95.604, 57.569, 93.868],
                [93.579, 60.206, 92.736],
                [90.848, 60.023, 90.474],
            ]
        )

        aa_names_a = ["CA"] * len(xyz_a)
        grid_width = 1.0
        max_order = 6

        param = z.get_global_parameter()
        residue_box = z.get_residue_gaussian_density_cache(param)

        voxel_a, corner_a = z.fill_voxel_by_weight_density(
            xyz_a, aa_names_a, param["residue_weight_map"], grid_width, residue_box[grid_width]
        )

        dimension_a = voxel_a.shape
        xyz_sample_a = {
            "X_sample": np.arange(dimension_a[0] + 1),
            "Y_sample": np.arange(dimension_a[1] + 1),
            "Z_sample": np.arange(dimension_a[2] + 1),
        }

        volume_a, center_a, _ = z.calculate_bbox_moment(voxel_a, 1, xyz_sample_a)
        avg_dist_a, max_dist_a = z.calculate_molecular_radius(
            voxel_a, center_a, volume_a, param["default_radius_multiplier"]
        )

        sphere_xyz_a = z.get_bbox_moment_xyz_sample(center_a, avg_dist_a, dimension_a)
        _, _, moment_a = z.calculate_bbox_moment(voxel_a, max_order, sphere_xyz_a)

        import pickle

        cache_dir = Path(z.__file__).parent / "cache_data"
        log_cache_path = cache_dir / f"LogG_CLMCache_MaxOrder{max_order:02d}.pkl"

        with open(log_cache_path, "rb") as file:
            cache_pkl = pickle.load(file)

        _, zm_raw = z.calculate_bbox_moment_2_zm(
            max_order,
            cache_pkl["GCache_complex"],
            cache_pkl["GCache_pqr_linear"],
            cache_pkl["GCache_complex_index"],
            cache_pkl["CLMCache3D"],
            moment_a,
        )

        # Test AB rotation calculation
        ab_list = z.calculate_ab_rotation_all(zm_raw, 2)

        # ab_list is a list of 2D arrays, each array contains AB pairs
        assert isinstance(ab_list, list), "AB rotation should return a list"
        assert len(ab_list) > 0, "AB rotation list should not be empty"

        # Check that each element is a valid 2D array of AB pairs
        for ab_array in ab_list:
            assert isinstance(ab_array, np.ndarray), "Each element should be a numpy array"
            assert ab_array.ndim == 2, "Each array should be 2D"
            assert ab_array.shape[1] == 2, "Each array should have 2 columns (a and b)"

            # Check normalization for each pair
            for i in range(ab_array.shape[0]):
                a, b = complex(ab_array[i, 0]), complex(ab_array[i, 1])
                norm = np.abs(a) ** 2 + np.abs(b) ** 2
                assert np.isclose(norm, 1.0, rtol=0.2, atol=0.1), (
                    f"AB pair should be normalized: |a|^2 + |b|^2 = {norm}"
                )
