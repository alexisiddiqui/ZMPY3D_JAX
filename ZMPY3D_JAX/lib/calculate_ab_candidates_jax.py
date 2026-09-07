from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp

from ZMPY3D_JAX.lib.eigen_root import _eigen_root_jax, batched_eigen_root


class ABRotationCandidates(NamedTuple):
    """Fixed-shape Cayley--Klein pairs and their validity mask."""

    pairs: chex.Array
    is_valid: chex.Array


def _candidate_complex_dtype(*values: chex.Array):
    """Preserve an explicit complex128 precision frontier when supplied."""
    return jnp.result_type(
        *(jnp.asarray(value).dtype for value in values), jnp.complex64
    )


def _compute_ab_candidates_impl(
    z_moment_raw: chex.Array, abconj_sol: chex.Array, ind_real: int
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """
    Compute ALL candidate a/b values with validity mask.
    Returns fixed-size arrays - filtering happens outside JIT.
    """
    complex_dtype = _candidate_complex_dtype(z_moment_raw, abconj_sol)
    z_moment_raw = jnp.asarray(z_moment_raw, dtype=complex_dtype)
    abconj_sol = jnp.asarray(abconj_sol, dtype=complex_dtype)
    k_re = jnp.real(abconj_sol)
    k_im = jnp.imag(abconj_sol)
    k_im2, k_re2 = k_im**2, k_re**2
    k_im3, k_im4, k_re4 = k_im**3, k_im**4, k_re**4

    # Extract moments at this order
    f20 = jnp.real(z_moment_raw[ind_real, 2, 0])
    f21 = z_moment_raw[ind_real, 2, 1]
    f22 = z_moment_raw[ind_real, 2, 2]
    f21_im, f21_re = jnp.imag(f21), jnp.real(f21)
    f22_im, f22_re = jnp.imag(f22), jnp.real(f22)

    # Vectorized coefficient calculation
    coef4 = (
        4 * f22_re * k_im * (-1 + k_im2 - 3 * k_re2)
        - 4 * f22_im * k_re * (1 - 3 * k_im2 + k_re2)
        - 2 * f21_re * k_im * k_re * (-3 + k_im2 + k_re2)
        + 2 * f20 * k_im * (-1 + k_im2 + k_re2)
        + f21_im * (1 - 6 * k_im2 + k_im2**2 - k_re2**2)
    )
    coef3 = 2 * (
        -4 * f22_im * (k_im + k_im3 - 3 * k_im * k_re2)
        + f21_re * (-1 + k_im4 + 6 * k_re2 - k_re4)
        + 2
        * k_re
        * (
            f22_re * (2 + 6 * k_im2 - 2 * k_re2)
            + f21_im * k_im * (-3 + k_im2 + k_re2)
            + f20 * (-1 + k_im2 + k_re2)
        )
    )

    # Stack coefficients and solve in batch
    bimbre_coef = jnp.stack([coef4, coef3, jnp.zeros_like(coef4), coef3, -coef4], axis=1)
    bimbre_sol = batched_eigen_root(bimbre_coef)
    bimbre_sol_real = jnp.real(bimbre_sol)

    # Compute ALL a/b values (fixed shape)
    k_im2_exp = k_im2[:, None]
    k_re2_exp = k_re2[:, None]
    abconj_sol_exp = abconj_sol[:, None]

    bre = 1 / jnp.sqrt((1 + k_im2_exp + k_re2_exp) * (1 + bimbre_sol_real**2))
    bim = bimbre_sol_real * bre
    b = jax.lax.complex(bre, bim)
    a = abconj_sol_exp * jnp.conj(b)

    # Compute validity mask but DON'T filter yet
    # Float32 eigensolves leave nominally-zero roots around 1e-7. Scale the
    # upstream cutoff by machine precision so those roots are not duplicated.
    root_tolerance = jnp.maximum(
        jnp.asarray(1e-7, dtype=bimbre_sol_real.dtype),
        jnp.asarray(100 * jnp.finfo(bimbre_sol_real.dtype).eps, dtype=bimbre_sol_real.dtype),
    )
    is_valid = jnp.abs(bimbre_sol_real) > root_tolerance

    # Return as numpy arrays for easier downstream processing
    return a, b, is_valid


def _secondary_coefficients(
    z_moment_raw: chex.Array, abconj_sol: chex.Array, ind_real: int
) -> tuple[chex.Array, chex.Array]:
    """Return the two independent coefficients of the secondary polynomial."""
    complex_dtype = _candidate_complex_dtype(z_moment_raw, abconj_sol)
    z_moment_raw = jnp.asarray(z_moment_raw, dtype=complex_dtype)
    abconj_sol = jnp.asarray(abconj_sol, dtype=complex_dtype)
    k_re = jnp.real(abconj_sol)
    k_im = jnp.imag(abconj_sol)
    k_im2, k_re2 = k_im**2, k_re**2
    k_im3, k_im4, k_re4 = k_im**3, k_im**4, k_re**4

    f20 = jnp.real(z_moment_raw[ind_real, 2, 0])
    f21 = z_moment_raw[ind_real, 2, 1]
    f22 = z_moment_raw[ind_real, 2, 2]
    f21_im, f21_re = jnp.imag(f21), jnp.real(f21)
    f22_im, f22_re = jnp.imag(f22), jnp.real(f22)

    coef4 = (
        4 * f22_re * k_im * (-1 + k_im2 - 3 * k_re2)
        - 4 * f22_im * k_re * (1 - 3 * k_im2 + k_re2)
        - 2 * f21_re * k_im * k_re * (-3 + k_im2 + k_re2)
        + 2 * f20 * k_im * (-1 + k_im2 + k_re2)
        + f21_im * (1 - 6 * k_im2 + k_im2**2 - k_re2**2)
    )
    coef3 = 2 * (
        -4 * f22_im * (k_im + k_im3 - 3 * k_im * k_re2)
        + f21_re * (-1 + k_im4 + 6 * k_re2 - k_re4)
        + 2
        * k_re
        * (
            f22_re * (2 + 6 * k_im2 - 2 * k_re2)
            + f21_im * k_im * (-3 + k_im2 + k_re2)
            + f20 * (-1 + k_im2 + k_re2)
        )
    )
    return coef4, coef3


def _compute_compact_ab_candidates_impl(
    z_moment_raw: chex.Array, abconj_sol: chex.Array, ind_real: int
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Solve only the two useful real roots of the secondary polynomial.

    The legacy quartic factors exactly as
    ``(t**2 + 1) * (coef4*t**2 + coef3*t - coef4)``.  Its two ``+/- i``
    roots are always rejected by the real-root validity mask, so retaining two
    fixed slots per initial root halves the subsequent rotation capacity.
    """
    complex_dtype = _candidate_complex_dtype(z_moment_raw, abconj_sol)
    z_moment_raw = jnp.asarray(z_moment_raw, dtype=complex_dtype)
    abconj_sol = jnp.asarray(abconj_sol, dtype=complex_dtype)
    coef4, coef3 = _secondary_coefficients(z_moment_raw, abconj_sol, ind_real)

    # Stable quadratic formula for a*t**2 + b*t - a.  The second root is
    # recovered from the product t1*t2=-1, avoiding cancellation.
    discriminant = jnp.sqrt(coef3**2 + 4 * coef4**2)
    sign_b = jnp.where(coef3 >= 0, 1, -1).astype(coef3.dtype)
    q = -0.5 * (coef3 + sign_b * discriminant)
    nondegenerate = (coef4 != 0) & (q != 0)
    safe_coef4 = jnp.where(nondegenerate, coef4, 1)
    safe_q = jnp.where(nondegenerate, q, 1)
    roots = jnp.stack((safe_q / safe_coef4, -safe_coef4 / safe_q), axis=1)

    root_tolerance = jnp.maximum(
        jnp.asarray(1e-7, dtype=roots.dtype),
        jnp.asarray(100 * jnp.finfo(roots.dtype).eps, dtype=roots.dtype),
    )
    initial_is_finite = jnp.isfinite(jnp.real(abconj_sol)) & jnp.isfinite(
        jnp.imag(abconj_sol)
    )
    is_valid = (
        nondegenerate[:, None]
        & initial_is_finite[:, None]
        & jnp.isfinite(roots)
        & (jnp.abs(roots) > root_tolerance)
    )

    k_norm2 = jnp.real(abconj_sol) ** 2 + jnp.imag(abconj_sol) ** 2
    bre = 1 / jnp.sqrt((1 + k_norm2[:, None]) * (1 + roots**2))
    bim = roots * bre
    b = jax.lax.complex(bre, bim)
    a = abconj_sol[:, None] * jnp.conj(b)

    # Invalid values must remain harmless because fixed-shape rotation evaluates
    # every slot before the masked reduction.
    a = jnp.where(is_valid, a, jnp.ones_like(a))
    b = jnp.where(is_valid, b, jnp.zeros_like(b))
    return a, b, is_valid


@partial(jax.jit, static_argnames=("ind_real",))
def compute_ab_candidates_jax(
    z_moment_raw: chex.Array, abconj_sol: chex.Array, ind_real: int
) -> tuple[chex.Array, chex.Array, chex.Array]:
    return _compute_ab_candidates_impl(z_moment_raw, abconj_sol, ind_real)


def _abconj_coefficients(
    z_moment_raw: chex.Array, target_order2_norm_rotate: int
) -> chex.Array:
    """Build the parity-specific polynomial coefficients inside a compiled trace."""
    if target_order2_norm_rotate % 2 == 0:
        return jnp.array(
            [
                z_moment_raw[target_order2_norm_rotate, 2, 2],
                -z_moment_raw[target_order2_norm_rotate, 2, 1],
                z_moment_raw[target_order2_norm_rotate, 2, 0],
                jnp.conj(z_moment_raw[target_order2_norm_rotate, 2, 1]),
                jnp.conj(z_moment_raw[target_order2_norm_rotate, 2, 2]),
            ],
            dtype=z_moment_raw.dtype,
        )
    return jnp.array(
        [
            z_moment_raw[target_order2_norm_rotate, 1, 1],
            -z_moment_raw[target_order2_norm_rotate, 1, 0],
            -jnp.conj(z_moment_raw[target_order2_norm_rotate, 1, 1]),
        ],
        dtype=z_moment_raw.dtype,
    )


def _stable_quadratic_roots(coefficients: chex.Array) -> chex.Array:
    """Solve one complex quadratic while avoiding the cancelled numerator."""
    coefficients = jnp.asarray(
        coefficients, dtype=_candidate_complex_dtype(coefficients)
    )
    a, b, c = coefficients
    nondegenerate = a != 0
    safe_a = jnp.where(nondegenerate, a, jnp.ones_like(a))
    discriminant = jnp.sqrt(b * b - 4 * a * c)
    numerator_plus = -b + discriminant
    numerator_minus = -b - discriminant
    numerator = jnp.where(
        jnp.abs(numerator_plus) >= jnp.abs(numerator_minus),
        numerator_plus,
        numerator_minus,
    )
    root1 = numerator / (2 * safe_a)
    root1_nonzero = root1 != 0
    safe_root1 = jnp.where(root1_nonzero, root1, jnp.ones_like(root1))
    root2 = c / (safe_a * safe_root1)
    valid = nondegenerate & root1_nonzero
    invalid = jnp.asarray(jnp.nan + 0j, dtype=coefficients.dtype)
    return jnp.where(valid, jnp.stack((root1, root2)), invalid)


def _initial_abconj_roots(
    coefficients: chex.Array,
    target_order2_norm_rotate: int,
    root_strategy: str,
) -> chex.Array:
    if root_strategy not in ("companion", "analytic_odd"):
        raise ValueError("root_strategy must be 'companion' or 'analytic_odd'")
    if root_strategy == "analytic_odd" and target_order2_norm_rotate % 2 == 1:
        return _stable_quadratic_roots(coefficients)
    return _eigen_root_jax(coefficients)


def _candidate_input(z_moment_raw: chex.Array, precision: str) -> chex.Array:
    if precision == "configured":
        from ZMPY3D_JAX import config as config

        return jnp.asarray(z_moment_raw, dtype=config.COMPLEX_DTYPE)
    if precision == "input":
        return jnp.asarray(
            z_moment_raw, dtype=_candidate_complex_dtype(z_moment_raw)
        )
    raise ValueError("precision must be 'configured' or 'input'")


@partial(jax.jit, static_argnums=(1, 2))
def calculate_ab_rotation_candidates(
    z_moment_raw: chex.Array,
    target_order2_norm_rotate: int,
    precision: str = "configured",
) -> ABRotationCandidates:
    """Generate the single-order candidates without dynamic filtering."""
    z_moment_raw = _candidate_input(z_moment_raw, precision)
    coefficients = _abconj_coefficients(z_moment_raw, target_order2_norm_rotate)
    abconj_sol = _eigen_root_jax(coefficients)
    a, b, is_valid = _compute_ab_candidates_impl(z_moment_raw, abconj_sol, 2)
    pairs = jnp.stack((a, b), axis=-1).reshape((-1, 2))
    return ABRotationCandidates(pairs, is_valid.reshape(-1))


@partial(jax.jit, static_argnums=(1, 2, 3))
def calculate_ab_rotation_compact_candidates(
    z_moment_raw: chex.Array,
    target_order2_norm_rotate: int,
    root_strategy: str = "companion",
    precision: str = "configured",
) -> ABRotationCandidates:
    """Generate only the two useful secondary roots per initial root."""
    z_moment_raw = _candidate_input(z_moment_raw, precision)
    coefficients = _abconj_coefficients(z_moment_raw, target_order2_norm_rotate)
    abconj_sol = _initial_abconj_roots(
        coefficients, target_order2_norm_rotate, root_strategy
    )
    a, b, is_valid = _compute_compact_ab_candidates_impl(
        z_moment_raw, abconj_sol, 2
    )
    pairs = jnp.stack((a, b), axis=-1).reshape((-1, 2))
    return ABRotationCandidates(pairs, is_valid.reshape(-1))


@partial(jax.jit, static_argnums=(1, 2, 3))
def calculate_ab_rotation_compact_candidate_group(
    z_moment_raw: chex.Array,
    target_orders: tuple[int, ...],
    root_strategy: str = "companion",
    precision: str = "configured",
) -> ABRotationCandidates:
    """Generate same-degree compact candidates in one batched eigensolve."""
    if not target_orders or len({order % 2 for order in target_orders}) != 1:
        raise ValueError("target_orders must be non-empty and share one parity")
    if root_strategy not in ("companion", "analytic_odd"):
        raise ValueError("root_strategy must be 'companion' or 'analytic_odd'")
    z_moment_raw = _candidate_input(z_moment_raw, precision)
    coefficients = jnp.stack(
        [_abconj_coefficients(z_moment_raw, order) for order in target_orders]
    )
    if root_strategy == "analytic_odd" and target_orders[0] % 2 == 1:
        roots = jax.vmap(_stable_quadratic_roots)(coefficients)
    else:
        roots = batched_eigen_root(coefficients)
    a, b, is_valid = jax.vmap(
        lambda item_roots: _compute_compact_ab_candidates_impl(
            z_moment_raw, item_roots, 2
        )
    )(roots)
    pairs = jnp.stack((a, b), axis=-1).reshape((len(target_orders), -1, 2))
    return ABRotationCandidates(pairs, is_valid.reshape((len(target_orders), -1)))


@partial(jax.jit, static_argnums=(1, 2))
def calculate_ab_rotation_all_candidates(
    z_moment_raw: chex.Array,
    target_order2_norm_rotate: int,
    precision: str = "configured",
) -> ABRotationCandidates:
    """Generate candidates for every supported ``ind_real`` as fixed groups."""
    z_moment_raw = _candidate_input(z_moment_raw, precision)
    coefficients = _abconj_coefficients(z_moment_raw, target_order2_norm_rotate)
    abconj_sol = _eigen_root_jax(coefficients)
    ind_real_all = jnp.arange(2, z_moment_raw.shape[0], 2)
    a, b, is_valid = jax.vmap(
        lambda ind_real: _compute_ab_candidates_impl(
            z_moment_raw, abconj_sol, ind_real
        )
    )(ind_real_all)
    group_count = ind_real_all.shape[0]
    pairs = jnp.stack((a, b), axis=-1).reshape((group_count, -1, 2))
    return ABRotationCandidates(pairs, is_valid.reshape((group_count, -1)))
