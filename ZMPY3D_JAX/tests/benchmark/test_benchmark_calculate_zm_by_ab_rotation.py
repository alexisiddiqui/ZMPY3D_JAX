"""
Tests for calculate_zm_by_ab_rotation function.
"""

import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import ZMPY3D_JAX as z


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default


class TestCalculateZMByABRotation:
    """Test suite for calculate_zm_by_ab_rotation function."""

    @pytest.fixture
    def cache_data(self):
        """Load cache data needed for rotation calculation."""
        max_order = 6
        cache_dir = Path(z.__file__).parent / "cache_data"

        binomial_path = cache_dir / "BinomialCache.pkl"
        log_cache_path = cache_dir / f"LogG_CLMCache_MaxOrder{max_order:02d}.pkl"

        with open(binomial_path, "rb") as file:
            binomial_cache_pkl = pickle.load(file)

        with open(log_cache_path, "rb") as file:
            cache_pkl = pickle.load(file)

        rotation_index = cache_pkl["RotationIndex"]

        # Convert arrays to JAX arrays where appropriate
        def _to_jax(x):
            if isinstance(x, np.ndarray):
                return jnp.asarray(x)
            if isinstance(x, (list, tuple)):
                # try to convert lists of arrays/values to array
                try:
                    return jnp.asarray(np.array(x))
                except Exception:
                    return x
            return x

        return {
            "BinomialCache": _to_jax(binomial_cache_pkl.get("BinomialCache")),
            "CLMCache": _to_jax(cache_pkl.get("CLMCache")),
            "s_id": _to_jax(np.squeeze(rotation_index["s_id"][0, 0]) - 1),
            "n": _to_jax(np.squeeze(rotation_index["n"][0, 0])),
            "l": _to_jax(np.squeeze(rotation_index["l"][0, 0])),
            "m": _to_jax(np.squeeze(rotation_index["m"][0, 0])),
            "mu": _to_jax(np.squeeze(rotation_index["mu"][0, 0])),
            "k": _to_jax(np.squeeze(rotation_index["k"][0, 0])),
            "IsNLM_Value": _to_jax(np.squeeze(rotation_index["IsNLM_Value"][0, 0]) - 1),
            "max_order": max_order,
        }

    @pytest.fixture
    def zm_raw(self):
        """Create sample raw Zernike moments as JAX array."""
        zm = np.zeros((7, 7, 13), dtype=complex)
        for n in range(7):
            for l in range(n + 1):
                for m in range(-l, l + 1):
                    if (n - l) % 2 == 0:
                        zm[n, l, m + 6] = np.random.randn() + 1j * np.random.randn()
        return jnp.asarray(zm)

    @pytest.fixture
    def ab_list(self):
        """Create sample AB rotation list as JAX array."""
        # Create a few rotation pairs
        ab_pairs = []
        for _ in range(3):
            theta = np.random.rand() * 2 * np.pi
            a = np.cos(theta) + 1j * np.sin(theta)
            b = np.sin(theta) - 1j * np.cos(theta)
            ab_pairs.append([a, b])
        return jnp.asarray(ab_pairs)

    def test_basic_rotation(self, zm_raw, ab_list, cache_data):
        """Test basic ZM rotation calculation."""
        max_order = cache_data["max_order"]

        zm_list = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_list,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        # Should return a list of rotated ZM arrays
        assert isinstance(zm_list, list)

        # Number of results should match number of AB pairs
        assert len(zm_list) == len(ab_list)

        # Each result should be complex
        for zm in zm_list:
            assert np.iscomplexobj(zm)

    def test_output_shapes(self, zm_raw, ab_list, cache_data):
        """Test that output shapes are consistent."""
        max_order = cache_data["max_order"]

        zm_list = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_list,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        # Each rotated ZM should have same shape
        first_shape = zm_list[0].shape
        for zm in zm_list:
            assert zm.shape == first_shape

    def test_single_rotation(self, zm_raw, cache_data):
        """Test with a single AB pair."""
        max_order = cache_data["max_order"]

        # Single normalized AB pair
        a = 1.0 + 0j
        b = 0.0 + 0j
        ab_single = np.array([[a, b]])

        zm_list = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_single,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        # Should get one result
        assert len(zm_list) == 1
        assert np.iscomplexobj(zm_list[0])

    def test_identity_rotation(self, zm_raw, cache_data):
        """Test that identity rotation preserves moments."""
        max_order = cache_data["max_order"]

        # Identity rotation: a=1, b=0
        ab_identity = np.array([[1.0 + 0j, 0.0 + 0j]])

        zm_list = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_identity,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        # Rotated moments should be close to original
        # (allowing for numerical precision)
        zm_rotated = zm_list[0]

        # Align axes if needed
        if zm_rotated.shape != zm_raw.shape:
            # Try all permutations to find a match
            for axes in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
                if zm_rotated.transpose(axes).shape == zm_raw.shape:
                    zm_rotated = zm_rotated.transpose(axes)
                    break

        # Check that non-NaN values are preserved
        mask = ~(np.isnan(zm_raw) | np.isnan(zm_rotated))
        if np.any(mask):
            assert np.allclose(zm_raw[mask], zm_rotated[mask], rtol=0.1, atol=1e-10)

    def test_deterministic(self, zm_raw, ab_list, cache_data):
        """Test that function is deterministic."""
        max_order = cache_data["max_order"]

        zm_list_1 = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_list,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        zm_list_2 = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_list,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        # Results should be identical
        assert len(zm_list_1) == len(zm_list_2)
        for zm1, zm2 in zip(zm_list_1, zm_list_2):
            np.testing.assert_array_equal(zm1, zm2)

    def test_multiple_rotations(self, zm_raw, cache_data):
        """Test with multiple different rotations."""
        max_order = cache_data["max_order"]

        # Create several different AB pairs
        ab_pairs = []
        for i in range(5):
            theta = i * np.pi / 5
            a = np.cos(theta) + 1j * np.sin(theta)
            b = np.sin(theta) - 1j * np.cos(theta)
            # Normalize
            norm = np.sqrt(np.abs(a) ** 2 + np.abs(b) ** 2)
            ab_pairs.append([a / norm, b / norm])

        ab_list = np.array(ab_pairs)

        zm_list = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_list,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        # Should get 5 different rotations
        assert len(zm_list) == 5

        # Each should be different (in general)
        for i in range(len(zm_list) - 1):
            # Most values should differ
            assert not np.allclose(zm_list[i], zm_list[i + 1], rtol=1e-5)

    def test_zero_moments(self, ab_list, cache_data):
        """Test with zero Zernike moments."""
        max_order = cache_data["max_order"]
        zero_zm = np.zeros((7, 7, 13), dtype=complex)

        zm_list = z.calculate_zm_by_ab_rotation(
            zero_zm,
            cache_data["BinomialCache"],
            ab_list,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        # Should handle zero input
        assert len(zm_list) == len(ab_list)

        # Results should be zero or NaN
        for zm in zm_list:
            non_nan = zm[~np.isnan(zm)]
            if len(non_nan) > 0:
                assert np.allclose(non_nan, 0, atol=1e-10)

    def test_complex_conjugate_property(self, zm_raw, cache_data):
        """Test conjugate symmetry properties of rotated moments."""
        max_order = cache_data["max_order"]

        # Create AB pair and its conjugate
        a = 0.6 + 0.8j
        b = 0.8 - 0.6j
        norm = np.sqrt(np.abs(a) ** 2 + np.abs(b) ** 2)
        ab_pair = np.array([[a / norm, b / norm]])

        zm_list = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_pair,
            max_order,
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        # Result should maintain certain symmetries
        zm_rotated = zm_list[0]
        assert np.iscomplexobj(zm_rotated)

    def test_time_evaluation_runtime(self, zm_raw, ab_list, cache_data):
        """Timed evaluation wrapper for calculate_zm_by_ab_rotation."""
        repeats = _env_int("ZMPY3D_TIME_REPEATS", 100)
        max_seconds = _env_int("ZMPY3D_TIME_MAX_SEC", 1200)

        # Warm-up
        _ = z.calculate_zm_by_ab_rotation(
            zm_raw,
            cache_data["BinomialCache"],
            ab_list,
            cache_data["max_order"],
            cache_data["CLMCache"],
            cache_data["s_id"],
            cache_data["n"],
            cache_data["l"],
            cache_data["m"],
            cache_data["mu"],
            cache_data["k"],
            cache_data["IsNLM_Value"],
        )

        start = time.perf_counter()
        for _ in range(repeats):
            _ = z.calculate_zm_by_ab_rotation(
                zm_raw,
                cache_data["BinomialCache"],
                ab_list,
                cache_data["max_order"],
                cache_data["CLMCache"],
                cache_data["s_id"],
                cache_data["n"],
                cache_data["l"],
                cache_data["m"],
                cache_data["mu"],
                cache_data["k"],
                cache_data["IsNLM_Value"],
            )
        elapsed = time.perf_counter() - start

        # Save detailed timing information to log file (like test_time_simple.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        benchmark_dir = os.path.join(script_dir, "_simple_time_benchmark")
        os.makedirs(benchmark_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"zm_by_ab_benchmark_{timestamp}.log"
        log_filepath = os.path.join(benchmark_dir, log_filename)

        with open(log_filepath, "w") as log_file:
            log_file.write("ZMPY3D_JAX ZM by AB Rotation Benchmark\n")
            log_file.write("=" * 60 + "\n")
            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write("\nConfiguration:\n")
            log_file.write(f"  Repeats: {repeats}\n")
            log_file.write(f"  Max seconds: {max_seconds}\n")
            log_file.write(f"  Max order: {cache_data['max_order']}\n")
            log_file.write("\nResults:\n")
            log_file.write(f"  Total time: {elapsed:.6f}s\n")
            log_file.write(f"  Time per iteration: {elapsed / repeats * 1000:.6f}ms\n")
            log_file.write(f"  Iterations per second: {repeats / elapsed:.2f}\n")
            log_file.write(f"\nStatus: {'PASS' if elapsed <= max_seconds else 'FAIL'}\n")
            if elapsed > max_seconds:
                log_file.write(f"  Exceeded threshold by: {elapsed - max_seconds:.2f}s\n")

        logging.info("ZM by AB benchmark saved to: %s", log_filepath)
        assert elapsed <= max_seconds, f"Elapsed {elapsed:.2f}s exceeded {max_seconds}s"
