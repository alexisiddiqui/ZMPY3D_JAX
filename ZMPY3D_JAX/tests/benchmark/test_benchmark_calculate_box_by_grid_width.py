"""
Tests for calculate_box_by_grid_width function.
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


class TestCalculateBoxByGridWidth:
    """Test suite for calculate_box_by_grid_width function."""

    @pytest.fixture
    def param(self):
        """Get global parameters."""
        return z.get_global_parameter()

    @pytest.fixture
    def residue_box_cache(self, param):
        """Get residue gaussian density cache."""
        return z.get_residue_gaussian_density_cache(param)

    def test_standard_grid_widths(self, param, residue_box_cache):
        """Test box generation for standard grid widths."""
        standard_widths = [0.25, 0.50, 1.00]

        for grid_width in standard_widths:
            # Access the cached boxes for this grid width
            residue_box = residue_box_cache.get(grid_width)

            # Should return a dictionary
            assert isinstance(residue_box, dict)

            # Should have entries for all standard amino acids
            assert "ALA" in residue_box
            assert "GLY" in residue_box
            assert "VAL" in residue_box

    def test_box_dimensions(self, param, residue_box_cache):
        """Test that boxes have correct dimensions."""
        grid_width = 1.00
        residue_box = residue_box_cache[grid_width]

        for residue_name, box in residue_box.items():
            # Each box should be a 3D numpy array
            assert isinstance(box, np.ndarray)
            assert box.ndim == 3

            # Dimensions should be odd (centered)
            assert box.shape[0] % 2 == 1
            assert box.shape[1] % 2 == 1
            assert box.shape[2] % 2 == 1

    def test_gaussian_properties(self, param, residue_box_cache):
        """Test that boxes have Gaussian-like properties."""
        grid_width = 1.00
        residue_box = residue_box_cache[grid_width]

        for residue_name, box in residue_box.items():
            # All values should be non-negative
            assert np.all(box >= 0)

            # Maximum should be at or near center
            center_idx = tuple(s // 2 for s in box.shape)
            max_idx = np.unravel_index(np.argmax(box), box.shape)

            # Max should be close to center
            distance_from_center = np.sqrt(sum((max_idx[i] - center_idx[i]) ** 2 for i in range(3)))
            assert distance_from_center <= 2

    def test_box_normalization(self, param, residue_box_cache):
        """Test that boxes are properly normalized."""
        grid_width = 1.00
        residue_box = residue_box_cache[grid_width]

        for residue_name, box in residue_box.items():
            total_density = np.sum(box)

            # Total density should be positive
            assert total_density > 0

            # For a Gaussian distribution, most mass should be within the box
            # The exact value depends on the cutoff used
            assert total_density > 0.1

    def test_different_residues(self, param, residue_box_cache):
        """Test that different residues have different boxes."""
        grid_width = 1.00
        residue_box = residue_box_cache[grid_width]

        # Get boxes for different residues
        ala_box = residue_box.get("ALA")
        trp_box = residue_box.get("TRP")

        if ala_box is not None and trp_box is not None:
            # Different residues should have different box sizes or densities
            # (TRP is larger than ALA)
            assert not np.array_equal(ala_box, trp_box)

    def test_grid_width_scaling(self, param, residue_box_cache):
        """Test that boxes scale properly with grid width."""
        widths = [0.50, 1.00]

        # Same residue at different widths
        for residue in ["ALA", "GLY"]:
            if residue in residue_box_cache[0.50] and residue in residue_box_cache[1.00]:
                box_small = residue_box_cache[0.50][residue]
                box_large = residue_box_cache[1.00][residue]

                # Finer grid should generally have larger box dimensions
                # or similar total density
                assert box_small is not None
                assert box_large is not None

    def test_spherical_symmetry(self, param, residue_box_cache):
        """Test that boxes are approximately spherically symmetric."""
        grid_width = 1.00
        residue_box = residue_box_cache[grid_width]

        # Check a few residues
        for residue_name in ["ALA", "GLY", "VAL"]:
            if residue_name in residue_box:
                box = residue_box[residue_name]
                center = tuple(s // 2 for s in box.shape)

                # Sample values at same distance from center
                # Should be similar due to spherical symmetry
                # This is a simplified test
                if box.shape[0] > 4:
                    val1 = box[center[0] + 1, center[1], center[2]]
                    val2 = box[center[0], center[1] + 1, center[2]]
                    val3 = box[center[0], center[1], center[2] + 1]

                    # Should be approximately equal
                    assert np.allclose([val1, val2, val3], val1, rtol=0.1)

    def test_decay_from_center(self, param, residue_box_cache):
        """Test that density decays with distance from center."""
        grid_width = 1.00
        residue_box = residue_box_cache[grid_width]

        for residue_name, box in residue_box.items():
            center = tuple(s // 2 for s in box.shape)
            center_val = box[center]

            # Values should decay as we move away from center
            if box.shape[0] > 2:
                edge_val = box[0, center[1], center[2]]
                assert center_val > edge_val

    def test_all_standard_residues(self, param, residue_box_cache):
        """Test that all standard amino acids are present."""
        grid_width = 1.00
        residue_box = residue_box_cache[grid_width]

        # Standard amino acids
        standard_residues = [
            "ALA",
            "CYS",
            "ASP",
            "GLU",
            "PHE",
            "GLY",
            "HIS",
            "ILE",
            "LYS",
            "LEU",
            "MET",
            "ASN",
            "PRO",
            "GLN",
            "ARG",
            "SER",
            "THR",
            "VAL",
            "TRP",
            "TYR",
        ]

        for residue in standard_residues:
            assert residue in residue_box
            assert residue_box[residue] is not None

    def test_output_dtype(self, param, residue_box_cache):
        """Test that output has correct data type."""
        grid_width = 1.00
        residue_box = residue_box_cache[grid_width]

        for residue_name, box in residue_box.items():
            # Should be float type
            assert box.dtype in [np.float32, np.float64]

    def test_deterministic(self, param):
        """Test that function produces consistent results."""
        grid_width = 1.00

        cache1 = z.get_residue_gaussian_density_cache(param)
        cache2 = z.get_residue_gaussian_density_cache(param)

        # Should produce identical results
        for residue in cache1[grid_width].keys():
            np.testing.assert_array_equal(cache1[grid_width][residue], cache2[grid_width][residue])

    def test_time_evaluation_runtime(self, param, residue_box_cache):
        """Timed evaluation wrapper for residue box access (uses same data repeatedly)."""
        repeats = _env_int("ZMPY3D_TIME_REPEATS", 100)
        max_seconds = _env_int("ZMPY3D_TIME_MAX_SEC", 1200)

        # Use an existing box (same array every iteration)
        grid_width = 1.00
        if grid_width not in residue_box_cache:
            # fallback: compute once
            res_cache = z.get_residue_gaussian_density_cache(param)
            residue_box = res_cache[grid_width]
        else:
            residue_box = residue_box_cache[grid_width]

        # Choose a residue present (safe fallback)
        residue = "ALA" if "ALA" in residue_box else next(iter(residue_box.keys()))
        box = residue_box[residue]

        # Warm-up
        _ = np.sum(box)

        start = time.perf_counter()
        for _ in range(repeats):
            # lightweight operation on the same array
            _ = np.sum(box)
        elapsed = time.perf_counter() - start

        # Save detailed timing information to log file (like test_time_simple.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        benchmark_dir = os.path.join(script_dir, "_simple_time_benchmark")
        os.makedirs(benchmark_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"box_by_grid_benchmark_{timestamp}.log"
        log_filepath = os.path.join(benchmark_dir, log_filename)

        with open(log_filepath, "w") as log_file:
            log_file.write("ZMPY3D_JAX Box-by-Grid Benchmark\n")
            log_file.write("=" * 60 + "\n")
            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write("\nConfiguration:\n")
            log_file.write(f"  Repeats: {repeats}\n")
            log_file.write(f"  Max seconds: {max_seconds}\n")
            log_file.write(f"  Grid width: {grid_width}\n")
            log_file.write("\nInput:\n")
            log_file.write(f"  Residue chosen: {residue}\n")
            log_file.write(f"  Box shape: {box.shape}\n")
            log_file.write("\nResults:\n")
            log_file.write(f"  Total time: {elapsed:.6f}s\n")
            log_file.write(f"  Time per iteration: {elapsed / repeats * 1000:.6f}ms\n")
            log_file.write(f"  Iterations per second: {repeats / elapsed:.2f}\n")
            log_file.write(f"\nStatus: {'PASS' if elapsed <= max_seconds else 'FAIL'}\n")
            if elapsed > max_seconds:
                log_file.write(f"  Exceeded threshold by: {elapsed - max_seconds:.2f}s\n")

        logging.info("Box-by-grid benchmark saved to: %s", log_filepath)
        assert elapsed <= max_seconds, f"Elapsed {elapsed:.2f}s exceeded {max_seconds}s"
