from functools import partial
from typing import List, NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

import ZMPY3D_JAX.config as _config

from .segmented_reduction import segmented_sum_associative


class ZMRotationCache(NamedTuple):
    """Device-resident constants used by the Zernike-moment rotation kernel."""

    binomial: chex.Array
    max_order: int
    clm: chex.Array
    s_id: chex.Array
    n: chex.Array
    l: chex.Array
    m: chex.Array
    mu: chex.Array
    k: chex.Array
    is_nlm_value: chex.Array


def prepare_zm_rotation_cache(
    binomial_cache: chex.Array,
    max_order: int,
    clm_cache: chex.Array,
    s_id: chex.Array,
    n: chex.Array,
    l: chex.Array,
    m: chex.Array,
    mu: chex.Array,
    k: chex.Array,
    is_nlm_value: chex.Array,
) -> ZMRotationCache:
    """Materialize static rotation data once using the configured JAX dtypes."""
    return ZMRotationCache(
        binomial=jnp.asarray(binomial_cache, dtype=_config.FLOAT_DTYPE),
        max_order=int(max_order),
        clm=jnp.asarray(clm_cache, dtype=_config.FLOAT_DTYPE),
        s_id=jnp.asarray(s_id, dtype=jnp.int32),
        n=jnp.asarray(n, dtype=jnp.int32),
        l=jnp.asarray(l, dtype=jnp.int32),
        m=jnp.asarray(m, dtype=jnp.int32),
        mu=jnp.asarray(mu, dtype=jnp.int32),
        k=jnp.asarray(k, dtype=jnp.int32),
        is_nlm_value=jnp.asarray(is_nlm_value, dtype=jnp.int32),
    )


_segmented_sum_associative = segmented_sum_associative


@partial(jax.jit, static_argnums=(3, 12))
def _calculate_zm_by_ab_rotation_jax(
    z_moment_raw: chex.Array,
    binomial_cache: chex.Array,
    ab_list: chex.Array,
    max_order: int,
    clm_cache: chex.Array,
    s_id: chex.Array,
    n: chex.Array,
    l: chex.Array,
    m: chex.Array,
    mu: chex.Array,
    k: chex.Array,
    is_nlm_value: chex.Array,
    reduction_strategy: str = "auto",
) -> chex.Array:
    """Vectorized rotation kernel with one output array per ``(a, b)`` pair."""
    complex_dtype = _config.COMPLEX_DTYPE
    float_dtype = _config.FLOAT_DTYPE

    z_moment_raw = jnp.asarray(z_moment_raw, dtype=complex_dtype)
    binomial_cache = jnp.asarray(binomial_cache, dtype=float_dtype)
    ab_list = jnp.asarray(ab_list, dtype=complex_dtype).reshape((-1, 2))
    clm_cache = jnp.asarray(clm_cache, dtype=float_dtype)
    s_id = jnp.asarray(s_id, dtype=jnp.int32)
    n = jnp.asarray(n, dtype=jnp.int32)
    l = jnp.asarray(l, dtype=jnp.int32)
    m = jnp.asarray(m, dtype=jnp.int32)
    mu = jnp.asarray(mu, dtype=jnp.int32)
    k = jnp.asarray(k, dtype=jnp.int32)
    is_nlm_value = jnp.asarray(is_nlm_value, dtype=jnp.int32)
    if reduction_strategy not in ("auto", "scatter", "segmented_scan"):
        raise ValueError(
            "reduction_strategy must be 'auto', 'scatter', or 'segmented_scan'"
        )
    use_segmented_scan = reduction_strategy == "segmented_scan" or (
        reduction_strategy == "auto" and complex_dtype == jnp.complex64
    )

    positive_mu = z_moment_raw[n, l, jnp.abs(mu)]
    negative_phase = jnp.where(mu % 2 == 0, 1.0, -1.0).astype(complex_dtype)
    f = jnp.where(mu >= 0, positive_mu, negative_phase * jnp.conj(positive_mu))
    nonzero_f = f != 0
    log_f = jnp.where(nonzero_f, jnp.log(f), jnp.zeros_like(f))

    max_n = max_order + 1
    clm = jnp.asarray(clm_cache[l * max_n + m], dtype=complex_dtype).reshape(-1)
    binomial = jnp.asarray(
        binomial_cache[l - mu, k - mu] + binomial_cache[l + mu, k - m],
        dtype=complex_dtype,
    )
    output_size = z_moment_raw.size
    nan_value = jnp.asarray(jnp.nan + 0j, dtype=complex_dtype)

    def format_output(z_nlm):
        flat = jnp.full((output_size,), nan_value, dtype=complex_dtype)
        flat = flat.at[is_nlm_value].set(z_nlm)
        return jnp.transpose(flat.reshape(z_moment_raw.shape), (2, 1, 0))

    def rotate_one(ab):
        a, b = ab
        identity = (jnp.abs(b) <= 1e-12) & (jnp.abs(jnp.imag(a)) <= 1e-12) & (
            jnp.abs(jnp.abs(jnp.real(a)) - 1.0) <= 1e-12
        )

        def identity_rotation(_):
            return jnp.transpose(z_moment_raw, (2, 1, 0))

        def ordinary_rotation(_):
            aac = jnp.asarray(jnp.real(a * jnp.conj(a)), dtype=complex_dtype)
            bbc = jnp.asarray(jnp.real(b * jnp.conj(b)), dtype=complex_dtype)
            bbcaac = -bbc / aac
            abc = -(a / jnp.conj(b))
            ab = a / b

            nlm = (
                log_f
                + jnp.log(aac) * l
                + clm
                + jnp.log(ab) * m
                + jnp.log(abc) * mu
                + jnp.log(bbcaac) * k
                + binomial
            )
            contributions = jnp.where(nonzero_f, jnp.exp(nlm), 0.0)
            if use_segmented_scan:
                z_nlm = _segmented_sum_associative(
                    contributions, s_id, is_nlm_value.shape[0]
                )
            else:
                z_nlm = jnp.zeros(is_nlm_value.shape, dtype=complex_dtype)
                z_nlm = z_nlm.at[s_id].add(contributions)
            return format_output(z_nlm)

        return jax.lax.cond(identity, identity_rotation, ordinary_rotation, operand=None)

    return jax.vmap(rotate_one)(ab_list)


def calculate_zm_by_ab_rotation_batch(
    z_moment_raw: chex.Array,
    ab_list: chex.Array,
    cache: ZMRotationCache,
    *,
    reduction_strategy: str = "auto",
) -> chex.Array:
    """Rotate pairs into ``(r, m, l, n)``; auto uses scan for float32."""
    return _calculate_zm_by_ab_rotation_jax(
        jnp.asarray(z_moment_raw, dtype=_config.COMPLEX_DTYPE),
        cache.binomial,
        jnp.asarray(ab_list, dtype=_config.COMPLEX_DTYPE).reshape((-1, 2)),
        cache.max_order,
        cache.clm,
        cache.s_id,
        cache.n,
        cache.l,
        cache.m,
        cache.mu,
        cache.k,
        cache.is_nlm_value,
        reduction_strategy,
    )


def calculate_zm_by_ab_rotation01(
    z_moment_raw: np.ndarray,
    binomial_cache: np.ndarray,
    ab_list: np.ndarray,
    max_order: int,
    clm_cache: np.ndarray,
    s_id: np.ndarray,
    n: np.ndarray,
    l: np.ndarray,
    m: np.ndarray,
    mu: np.ndarray,
    k: np.ndarray,
    is_nlm_value: np.ndarray,
) -> List[np.ndarray]:
    """Rotate raw Zernike moments using Cayley--Klein ``(a, b)`` pairs.

    The public API retains the upstream list-of-NumPy-arrays return type. Internally,
    rotations are evaluated together by a vectorized JAX kernel. Output axes retain
    the upstream ``(m, l, n)`` layout.
    """
    cache = prepare_zm_rotation_cache(
        binomial_cache,
        max_order,
        clm_cache,
        s_id,
        n,
        l,
        m,
        mu,
        k,
        is_nlm_value,
    )
    rotated = calculate_zm_by_ab_rotation_batch(z_moment_raw, ab_list, cache)
    return [np.asarray(item) for item in rotated]
