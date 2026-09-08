"""Stable Cartesian recurrence for orthonormal three-dimensional Zernike modes.

``basis`` evaluates ``conj(B_nlm)`` where

``B_nlm = sqrt(2*n + 3) r**l P_k^(0,l+1/2)(2*r**2-1) Y_l^m``.

SciPy's Condon--Shortley convention is used for ``Y``.  The conjugation is the
one required by the legacy coefficient tables; the remaining legacy phase and
normalisation are supplied by :mod:`direct_moment_backend`.  The recurrences
are polynomial in Cartesian coordinates, so the origin, axes, and points
outside the unit ball require no special angular handling.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np


def mode_indices(max_order: int) -> np.ndarray:
    """Return structurally present ``(n, l, m)`` modes in deterministic order."""
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    return np.asarray(
        [
            (n, ell, m)
            for n in range(max_order + 1)
            for ell in range(n + 1)
            if (n - ell) % 2 == 0
            for m in range(ell + 1)
        ],
        dtype=np.int32,
    )


def _jacobi(k: int, beta: float, x, dtype):
    """Evaluate ``P_k^(0,beta)(x)`` with the three-term recurrence."""
    x = jnp.asarray(x, dtype=dtype)
    if k == 0:
        return jnp.ones_like(x)
    p0 = jnp.ones_like(x)
    p1 = ((-beta) + (beta + 2.0) * x) / 2.0
    if k == 1:
        return p1
    for degree in range(2, k + 1):
        d = jnp.asarray(degree, dtype=dtype)
        two = 2.0 * d + beta
        a = 2.0 * d * (d + beta) * (two - 2.0)
        b = (two - 1.0) * (-beta * beta + two * (two - 2.0) * x)
        c = 2.0 * (d - 1.0) * (d + beta - 1.0) * two
        p0, p1 = p1, (b * p1 - c * p0) / a
    return p1


def _solid_harmonics(points, max_order: int, dtype):
    """Return Cartesian ``r**l * conj(Y_l^m)`` as real/imaginary arrays."""
    points = jnp.asarray(points, dtype=dtype)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r2 = x * x + y * y + z * z
    values: dict[tuple[int, int], object] = {}
    complex_dtype = jnp.complex128 if jnp.dtype(dtype) == jnp.float64 else jnp.complex64
    xy = jnp.asarray(x - 1j * y, dtype=complex_dtype)
    for m in range(max_order + 1):
        double_factorial = math.prod(range(1, 2 * m, 2)) if m else 1
        s_mm = jnp.asarray(((-1) ** m) * float(double_factorial), dtype=dtype) * xy**m
        norm = math.sqrt(
            (2 * m + 1) / (4 * math.pi) * math.factorial(0) / math.factorial(2 * m)
        )
        values[m, m] = s_mm * norm
        if m == max_order:
            continue
        s_prev2 = s_mm
        s_prev1 = (2 * m + 1) * z * s_mm
        norm = math.sqrt(
            (2 * (m + 1) + 1)
            / (4 * math.pi)
            * math.factorial(1)
            / math.factorial(2 * m + 1)
        )
        values[m + 1, m] = s_prev1 * norm
        for ell in range(m + 2, max_order + 1):
            s = ((2 * ell - 1) * z * s_prev1 - (ell + m - 1) * r2 * s_prev2) / (
                ell - m
            )
            norm = math.sqrt(
                (2 * ell + 1)
                / (4 * math.pi)
                * math.factorial(ell - m)
                / math.factorial(ell + m)
            )
            values[ell, m] = s * norm
            s_prev2, s_prev1 = s_prev1, s
    ordered = [values[ell, m] for ell in range(max_order + 1) for m in range(ell + 1)]
    result = jnp.stack(ordered, axis=1)
    return result.real, result.imag


def basis(points, max_order: int, *, dtype=jnp.float32):
    """Evaluate all valid orthonormal modes, returning ``(indices, values)``."""
    points = jnp.asarray(points, dtype=dtype)
    r2 = jnp.sum(points * points, axis=1)
    solid_real, solid_imag = _solid_harmonics(points, max_order, dtype)
    solid_offset = {(ell, m): ell * (ell + 1) // 2 + m for ell in range(max_order + 1) for m in range(ell + 1)}
    values = []
    indices = mode_indices(max_order)
    for n, ell, m in indices.tolist():
        radial = _jacobi((n - ell) // 2, ell + 0.5, 2.0 * r2 - 1.0, dtype)
        radial = radial * math.sqrt(2 * n + 3)
        pos = solid_offset[ell, m]
        values.append(radial * (solid_real[:, pos] + 1j * solid_imag[:, pos]))
    return jnp.asarray(indices, dtype=jnp.int32), jnp.stack(values, axis=1)


def pairwise_sum(values, axis: int = 0):
    """Deterministically sum after zero-padding the reduced axis to a power of two."""
    values = jnp.asarray(values)
    size = values.shape[axis]
    target = 1 if size == 0 else 1 << (size - 1).bit_length()
    padding = [(0, 0)] * values.ndim
    padding[axis] = (0, target - size)
    work = jnp.pad(values, padding)
    while work.shape[axis] > 1:
        shape = list(work.shape)
        shape[axis : axis + 1] = [shape[axis] // 2, 2]
        work = work.reshape(shape).sum(axis=axis + 1)
    return jnp.squeeze(work, axis=axis)
