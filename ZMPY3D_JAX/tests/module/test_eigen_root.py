"""
Tests for eigen_root function.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import ZMPY3D_JAX as z


class TestEigenRoot:
    """Test suite for eigen_root function."""

    def test_quadratic_polynomial_real_roots(self):
        """Test with a quadratic polynomial: x^2 - 3x + 2 = 0, roots = 1, 2."""
        # Coefficients for x^2 - 3x + 2 = 0 are [1, -3, 2]
        coefficients = np.array([1, -3, 2])
        roots = z.eigen_root(coefficients)
        assert len(roots) == 2
        assert np.allclose(sorted(roots), sorted([1.0, 2.0]))

    def test_quadratic_polynomial_complex_roots(self):
        """Test with a quadratic polynomial: x^2 + 1 = 0, roots = i, -i."""
        # Coefficients for x^2 + 0x + 1 = 0 are [1, 0, 1]
        coefficients = np.array([1, 0, 1])
        roots = z.eigen_root(coefficients)
        assert len(roots) == 2
        # Use a consistent sorting key that rounds real/imaginary parts for robustness
        assert np.allclose(sorted(roots, key=lambda x: (round(x.real, 10), round(x.imag, 10))),
                           sorted([1j, -1j], key=lambda x: (round(x.real, 10), round(x.imag, 10))),
                           atol=1e-10)

    def test_cubic_polynomial(self):
        """Test with a cubic polynomial: x^3 - 6x^2 + 11x - 6 = 0, roots = 1, 2, 3."""
        # Coefficients for x^3 - 6x^2 + 11x - 6 = 0 are [1, -6, 11, -6]
        coefficients = np.array([1, -6, 11, -6])
        roots = z.eigen_root(coefficients)
        assert len(roots) == 3
        assert np.allclose(sorted(roots), sorted([1.0, 2.0, 3.0]))

    def test_output_type(self):
        """Test that the output roots are complex numbers."""
        coefficients = np.array([1, 0, 1])
        roots = z.eigen_root(coefficients)
        assert all(isinstance(r, (complex, np.complexfloating)) for r in roots)

    def test_deterministic(self):
        """Test that the function is deterministic."""
        coefficients = np.array([1, -3, 2])
        roots1 = z.eigen_root(coefficients)
        roots2 = z.eigen_root(coefficients)
        assert np.allclose(sorted(roots1), sorted(roots2))

    def test_polynomial_with_repeated_roots(self):
        """Test with a polynomial having repeated roots: x^2 - 2x + 1 = 0, roots = 1, 1."""
        # Coefficients for x^2 - 2x + 1 = 0 are [1, -2, 1]
        coefficients = np.array([1, -2, 1])
        roots = z.eigen_root(coefficients)
        assert len(roots) == 2
        assert np.allclose(sorted(roots), sorted([1.0, 1.0]))

    def test_high_order_polynomial(self):
        """Test with a higher order polynomial."""
        # (x-1)(x-2)(x-3)(x-4)(x-5) = x^5 - 15x^4 + 85x^3 - 225x^2 + 274x - 120 = 0
        # Coefficients are [1, -15, 85, -225, 274, -120]
        coefficients = np.array([1, -15, 85, -225, 274, -120])
        roots = z.eigen_root(coefficients)
        assert len(roots) == 5
        assert np.allclose(sorted(roots), sorted([1.0, 2.0, 3.0, 4.0, 5.0]))
