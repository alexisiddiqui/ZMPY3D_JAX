"""
Tests for eigen_root function.
"""

import sys
from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import ZMPY3D_JAX as z

z.configure_for_scientific_computing()
print(f"JAX FLOAT_DTYPE after configuration: {z.FLOAT_DTYPE}")
print(f"JAX COMPLEX_DTYPE after configuration: {z.COMPLEX_DTYPE}")


class TestEigenRoot:
    """Test suite for eigen_root function."""

    def test_quadratic_polynomial_real_roots(self):
        """Test with a quadratic polynomial: x^2 - 3x + 2 = 0, roots = 1, 2."""
        # Coefficients for x^2 - 3x + 2 = 0 are [1, -3, 2]
        coefficients = np.array([1, -3, 2])
        roots = z.eigen_root(coefficients)
        roots_jnp = jnp.asarray(roots)
        chex.assert_shape(roots_jnp, (2,))
        expected = jnp.array([1.0, 2.0], dtype=roots_jnp.dtype)
        chex.assert_trees_all_close(jnp.sort(roots_jnp), jnp.sort(expected), atol=1e-8)

    def test_quadratic_polynomial_complex_roots(self):
        """Test with a quadratic polynomial: x^2 + 1 = 0, roots = i, -i."""
        # Coefficients for x^2 + 0x + 1 = 0 are [1, 0, 1]
        coefficients = np.array([1, 0, 1])
        roots = z.eigen_root(coefficients)
        roots_jnp = jnp.asarray(roots)
        chex.assert_shape(roots_jnp, (2,))
        expected = jnp.array([1j, -1j], dtype=roots_jnp.dtype)
        # sort both and compare with a tight tolerance
        # Sort by imaginary part first, then real part, to handle cases where
        # the real part is non-zero due to floating point inaccuracies.
        sorted_roots = roots_jnp[jnp.lexsort((roots_jnp.real, roots_jnp.imag))]
        sorted_expected = expected[jnp.lexsort((expected.real, expected.imag))]
        chex.assert_trees_all_close(sorted_roots, sorted_expected, atol=1e-10)

    def test_cubic_polynomial(self):
        """Test with a cubic polynomial: x^3 - 6x^2 + 11x - 6 = 0, roots = 1, 2, 3."""
        # Coefficients for x^3 - 6x^2 + 11x - 6 = 0 are [1, -6, 11, -6]
        coefficients = np.array([1, -6, 11, -6])
        roots = z.eigen_root(coefficients)
        roots_jnp = jnp.asarray(roots)
        chex.assert_shape(roots_jnp, (3,))
        expected = jnp.array([1.0, 2.0, 3.0], dtype=roots_jnp.dtype)
        chex.assert_trees_all_close(jnp.sort(roots_jnp), jnp.sort(expected), atol=1e-8)

    def test_output_type(self):
        """Test that the output roots are complex numbers."""
        coefficients = np.array([1, 0, 1])
        roots = z.eigen_root(coefficients)
        roots_jnp = jnp.asarray(roots)
        # Ensure the returned array is a complex dtype
        chex.assert_equal(bool(jnp.iscomplexobj(roots_jnp)), True)

    def test_deterministic(self):
        """Test that the function is deterministic."""
        coefficients = np.array([1, -3, 2])
        roots1 = jnp.asarray(z.eigen_root(coefficients))
        roots2 = jnp.asarray(z.eigen_root(coefficients))
        chex.assert_trees_all_close(jnp.sort(roots1), jnp.sort(roots2), atol=1e-12)

    def test_polynomial_with_repeated_roots(self):
        """Test with a polynomial having repeated roots: x^2 - 2x + 1 = 0, roots = 1, 1."""
        # Coefficients for x^2 - 2x + 1 = 0 are [1, -2, 1]
        coefficients = np.array([1, -2, 1])
        roots = z.eigen_root(coefficients)
        roots_jnp = jnp.asarray(roots)
        chex.assert_shape(roots_jnp, (2,))
        expected = jnp.array([1.0, 1.0], dtype=roots_jnp.dtype)
        chex.assert_trees_all_close(jnp.sort(roots_jnp), jnp.sort(expected), atol=1e-8)

    def test_high_order_polynomial(self):
        """Test with a higher order polynomial."""
        # (x-1)(x-2)(x-3)(x-4)(x-5) = x^5 - 15x^4 + 85x^3 - 225x^2 + 274x - 120 = 0
        # Coefficients are [1, -15, 85, -225, 274, -120]
        coefficients = np.array([1, -15, 85, -225, 274, -120])
        roots = z.eigen_root(coefficients)
        roots_jnp = jnp.asarray(roots)
        chex.assert_shape(roots_jnp, (5,))
        expected = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=roots_jnp.dtype)
        chex.assert_trees_all_close(jnp.sort(roots_jnp), jnp.sort(expected), atol=1e-8)
