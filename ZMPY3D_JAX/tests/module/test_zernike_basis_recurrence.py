import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
scipy = pytest.importorskip("scipy.special")
jax.config.update("jax_enable_x64", True)

from ZMPY3D_JAX.lib.zernike_basis_recurrence import basis, pairwise_sum


def _reference(points, n, ell, m):
    r = np.linalg.norm(points, axis=1)
    theta = np.arccos(np.divide(points[:, 2], r, out=np.ones_like(r), where=r != 0))
    phi = np.arctan2(points[:, 1], points[:, 0])
    harmonic = scipy.sph_harm_y(ell, m, theta, phi)
    return (
        np.sqrt(2 * n + 3)
        * r**ell
        * scipy.eval_jacobi((n - ell) // 2, 0, ell + 0.5, 2 * r**2 - 1)
        * np.conj(harmonic)
    )


@pytest.mark.parametrize("max_order", [6, 20])
def test_basis_agrees_with_independent_scipy_reference(max_order):
    points = np.array([[0, 0, 0], [0, 0, 0.7], [.2, -.3, .4], [1.1, .2, -.4]])
    indices, actual = basis(points, max_order, dtype=jnp.float64)
    actual = np.asarray(actual)
    for column, (n, ell, m) in enumerate(np.asarray(indices)):
        np.testing.assert_allclose(actual[:, column], _reference(points, n, ell, m), rtol=2e-10, atol=2e-10)


def test_spherical_harmonic_reference_helper_low_order():
    point = np.array([[1.0, 0.0, 0.0]])
    expected = -np.sqrt(15 / (8 * np.pi))
    np.testing.assert_allclose(_reference(point, 1, 1, 1), expected)


def test_pairwise_sum_handles_non_power_of_two_and_is_repeatable():
    values = jnp.arange(21, dtype=jnp.float32).reshape(7, 3)
    first = pairwise_sum(values)
    np.testing.assert_array_equal(first, pairwise_sum(values))
    np.testing.assert_allclose(first, np.asarray(values).sum(axis=0))
