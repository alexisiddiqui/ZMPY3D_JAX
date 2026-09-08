"""Portable direct-moment benchmark; run explicitly with ``-m benchmark``."""
import json
import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ZMPY3D_JAX as z
from ZMPY3D_JAX.lib.calculate_bbox_moment_2_zm05 import BBoxToZMCache
from ZMPY3D_JAX.lib.mixed_precision_prototype import (
    calculate_bbox_moments_mixed_prototype,
    calculate_zm_mixed_prototype,
)
from ZMPY3D_JAX.lib.batched_descriptor import (
    _calculate_bbox_max_order_batch,
    _calculate_bbox_to_zm_batch,
)


def _measure(runner, argument, repeats):
    start = time.perf_counter()
    first = runner(argument)
    jax.block_until_ready(first)
    compile_seconds = time.perf_counter() - start
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = runner(argument)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return first, compile_seconds, samples


@pytest.mark.benchmark
def test_benchmark_direct_moments():
    backend = os.environ.get("ZMPY3D_BENCHMARK_BACKEND", "cpu").lower()
    if backend not in {"cpu", "gpu", "mps"}:
        pytest.fail("ZMPY3D_BENCHMARK_BACKEND must be cpu, gpu, or mps")
    z.configure_for_scientific_computing(enable_x64=False, platform=backend)
    order = int(os.environ.get("ZMPY3D_BENCHMARK_ORDER", "6"))
    cache_path = Path(z.__file__).parent / "cache_data" / f"LogG_CLMCache_MaxOrder{order:02d}.pkl"
    with cache_path.open("rb") as handle:
        data = pickle.load(handle)
    direct = z.prepare_direct_moment_cache(order, data["CLMCache3D"])
    legacy = z.prepare_bbox_to_zm_cache(order, data["GCache_complex"], data["GCache_pqr_linear"], data["GCache_complex_index"], data["CLMCache3D"])
    rng = np.random.default_rng(42)
    grid_size = int(os.environ.get("ZMPY3D_BENCHMARK_GRID_SIZE", "4" if order >= 20 else "8"))
    shape = (1, grid_size, grid_size, grid_size)
    voxels = (rng.random(shape) > .8) * rng.random(shape)
    edges = np.linspace(-.8, .8, grid_size + 1, dtype=np.float32)[None, :]

    repeats = int(os.environ.get("ZMPY3D_BENCHMARK_REPEATS", "7"))
    direct_runner = jax.jit(
        lambda v: z.calculate_direct_moments(v, edges, edges, edges, direct)
    )
    (scaled, raw), direct_compile, direct_samples = _measure(
        direct_runner, voxels, repeats
    )
    direct_steady = float(np.median(direct_samples))
    output = Path(__file__).parent / "_simple_time_benchmark"
    output.mkdir(exist_ok=True)
    if backend == "mps":
        report = {
            "backend": backend,
            "mps_async_dispatch": os.environ.get("JAX_MPS_ASYNC_DISPATCH") == "1",
            "order": order,
            "voxel_shape": list(voxels.shape),
            "occupied_cells": int(np.count_nonzero(voxels)),
            "quadrature_samples_per_cell": int(direct.quadrature_weights.size),
            "repeats": repeats,
            "cartesian_mixed_moments_x64": {
                "available": False,
                "reason": "MPS does not support the float64 moments_x64 frontier",
            },
            "direct_recurrence": {
                "compile_seconds": direct_compile,
                "steady_moment_seconds": direct_steady,
                "samples_seconds": direct_samples,
            },
        }
        (output / f"moment_backends_{backend}_order{order}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2))
        return

    strict_cache = BBoxToZMCache(
        order,
        jnp.asarray(data["GCache_complex"].reshape(-1), dtype=jnp.complex128),
        jnp.asarray(data["GCache_pqr_linear"].reshape(-1) - 1, dtype=jnp.int32),
        jnp.asarray(data["GCache_complex_index"].reshape(-1) - 1, dtype=jnp.int32),
        jnp.asarray(data["CLMCache3D"], dtype=jnp.complex128),
    )

    def cartesian_moments(v):
        bbox = calculate_bbox_moments_mixed_prototype(
            v, order, edges, edges, edges, "moments_x64"
        )
        return calculate_zm_mixed_prototype(
            bbox,
            order,
            legacy.g_coefficients,
            legacy.pqr_indices,
            legacy.output_indices,
            legacy.clm,
            strict_cache.g_coefficients,
            strict_cache.clm,
            "moments_x64",
        )

    cartesian_runner = jax.jit(cartesian_moments)
    (reference_scaled, reference_raw), cartesian_compile, cartesian_samples = _measure(
        cartesian_runner, voxels, repeats
    )
    strict_voxels = jnp.asarray(voxels, dtype=jnp.float64)
    strict_edges = jnp.asarray(edges, dtype=jnp.float64)
    _, _, strict_bbox = _calculate_bbox_max_order_batch(
        strict_voxels, order, strict_edges, strict_edges, strict_edges
    )
    _, strict_raw = _calculate_bbox_to_zm_batch(
        strict_bbox, order, strict_cache.g_coefficients, strict_cache.pqr_indices,
        strict_cache.output_indices, strict_cache.clm, "segmented_scan"
    )
    actual = np.asarray(raw[0])
    cartesian_actual = np.asarray(reference_raw[0])
    reference = np.asarray(strict_raw[0])
    mask = np.isfinite(reference)
    def errors(value):
        delta = value - reference
        return {
            "relative_l2_vs_float64": float(np.linalg.norm(delta[mask]) / np.linalg.norm(reference[mask])),
            "max_component_error_vs_float64": float(np.max(np.abs(delta[mask]))),
        }
    cartesian_steady = float(np.median(cartesian_samples))
    report = {
        "backend": backend,
        "order": order,
        "voxel_shape": list(voxels.shape),
        "occupied_cells": int(np.count_nonzero(voxels)),
        "quadrature_samples_per_cell": int(direct.quadrature_weights.size),
        "repeats": repeats,
        "cartesian_mixed_moments_x64": {
            "compile_seconds": cartesian_compile,
            "steady_moment_seconds": cartesian_steady,
            "samples_seconds": cartesian_samples,
            **errors(cartesian_actual),
        },
        "direct_recurrence": {
            "compile_seconds": direct_compile,
            "steady_moment_seconds": direct_steady,
            "samples_seconds": direct_samples,
            **errors(actual),
        },
        "direct_over_cartesian_steady_ratio": direct_steady / cartesian_steady,
        "direct_vs_cartesian_relative_l2": float(np.linalg.norm((actual-cartesian_actual)[mask]) / np.linalg.norm(reference[mask])),
        "validity_masks_equal": bool(np.array_equal(np.isnan(actual), np.isnan(cartesian_actual))),
    }
    (output / f"moment_backends_{backend}_order{order}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
