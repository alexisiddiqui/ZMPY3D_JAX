"""Informational timing for internal order-20 mixed-precision prototypes."""

from __future__ import annotations

import json
import os
import pickle
import platform
import time
from pathlib import Path
from statistics import median

import jax
import jax.numpy as jnp
import jaxlib
import pytest

import ZMPY3D_JAX as z


BACKEND = os.getenv("ZMPY3D_BENCHMARK_BACKEND", "cpu").lower()
if BACKEND not in ("cpu", "gpu"):
    raise ValueError("ZMPY3D_BENCHMARK_BACKEND must be 'cpu' or 'gpu'")
z.configure_for_scientific_computing(enable_x64=False, platform=BACKEND)
jax.config.update("jax_enable_x64", True)

from ZMPY3D_JAX.ZMPY3D_CLI_BatchZM import (  # noqa: E402
    _prepare_batch_runtime,
)
from ZMPY3D_JAX.lib.batched_descriptor import (  # noqa: E402
    _calculate_bbox_max_order_batch,
    _calculate_bbox_order1_batch,
    _calculate_bbox_to_zm_batch,
    _calculate_radius_and_samples_batch,
    calculate_descriptor_batch_from_voxels,
    pad_voxel_batch,
)
from ZMPY3D_JAX.lib.calculate_bbox_moment_2_zm05 import (  # noqa: E402
    BBoxToZMCache,
)
from ZMPY3D_JAX.lib.fill_voxel_by_weight_density04 import (  # noqa: E402
    fill_voxel_by_weight_density_host,
)
from ZMPY3D_JAX.lib.mixed_precision_prototype import (  # noqa: E402
    PRECISION_FRONTIERS,
    calculate_bbox_moments_mixed_prototype,
    calculate_descriptor_from_voxels_mixed_prototype,
    calculate_zm_mixed_prototype,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MAX_ORDER = 20
TARGET_ORDERS = (2, 3, 4, 5)


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _block(value):
    return jax.block_until_ready(value)


def _samples(function, repeats: int, sample_count: int) -> list[float]:
    _block(function())
    samples = []
    for _ in range(sample_count):
        start = time.perf_counter_ns()
        for _ in range(repeats):
            result = function()
        _block(result)
        samples.append((time.perf_counter_ns() - start) / repeats / 1e9)
    return samples


def _first_execution(function):
    start = time.perf_counter_ns()
    result = function()
    _block(result)
    return result, (time.perf_counter_ns() - start) / 1e9


def _summary(samples: list[float], protein_count: int) -> dict[str, object]:
    median_seconds = median(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": median_seconds,
        "median_milliseconds_per_protein": median_seconds
        / protein_count
        * 1000.0,
    }


def _prepare_x64_cache() -> BBoxToZMCache:
    with (
        REPO_ROOT
        / "ZMPY3D_JAX"
        / "cache_data"
        / "LogG_CLMCache_MaxOrder20.pkl"
    ).open("rb") as handle:
        cache = pickle.load(handle)
    return BBoxToZMCache(
        max_order=MAX_ORDER,
        g_coefficients=jnp.asarray(
            cache["GCache_complex"], dtype=jnp.complex128
        ).reshape(-1),
        pqr_indices=jnp.asarray(
            cache["GCache_pqr_linear"], dtype=jnp.int32
        ).reshape(-1)
        - 1,
        output_indices=jnp.asarray(
            cache["GCache_complex_index"], dtype=jnp.int32
        ).reshape(-1)
        - 1,
        clm=jnp.asarray(cache["CLMCache3D"], dtype=jnp.complex128),
    )


@pytest.mark.benchmark
def test_mixed_precision_order20_snapshot() -> None:
    repeats = _positive_env_int("ZMPY3D_MIXED_REPEATS", 3)
    sample_count = _positive_env_int("ZMPY3D_MIXED_SAMPLES", 7)
    runtime = _prepare_batch_runtime(1.0, MAX_ORDER)
    x64_cache = _prepare_x64_cache()
    host_voxels = []
    for name in ("6NT5.pdb", "6NT6.pdb"):
        xyz, residues = z.get_pdb_xyz_ca(str(REPO_ROOT / name))
        voxel, _ = fill_voxel_by_weight_density_host(
            xyz,
            residues,
            runtime.param["residue_weight_map"],
            1.0,
            runtime.residue_box[1.0],
        )
        host_voxels.append(voxel)
    voxels = jnp.asarray(pad_voxel_batch(host_voxels), dtype=jnp.float32)
    masses, centers, _ = _calculate_bbox_order1_batch(voxels)
    radius = _calculate_radius_and_samples_batch(
        voxels,
        centers,
        masses,
        runtime.param["default_radius_multiplier"],
    )

    def baseline_bbox():
        return _calculate_bbox_max_order_batch(
            voxels, MAX_ORDER, radius[3], radius[4], radius[5]
        )[2]

    baseline_bbox_value, baseline_bbox_first = _first_execution(baseline_bbox)

    def baseline_zm():
        return _calculate_bbox_to_zm_batch(
            baseline_bbox_value,
            MAX_ORDER,
            runtime.bbox_to_zm_cache.g_coefficients,
            runtime.bbox_to_zm_cache.pqr_indices,
            runtime.bbox_to_zm_cache.output_indices,
            runtime.bbox_to_zm_cache.clm,
        )

    def baseline_descriptor():
        return calculate_descriptor_batch_from_voxels(
            voxels,
            max_order=MAX_ORDER,
            max_target_order=5,
            mode=2,
            default_radius_multiplier=runtime.param[
                "default_radius_multiplier"
            ],
            bbox_to_zm_cache=runtime.bbox_to_zm_cache,
            rotation_cache=runtime.rotation_cache,
            descriptor_cache=runtime.descriptor_cache,
            moment_precision="configured",
        )

    _, baseline_zm_first = _first_execution(baseline_zm)
    _, baseline_descriptor_first = _first_execution(baseline_descriptor)

    profiles = {
        "configured_float32": {
            "first_execution_seconds": {
                "bbox_max_order": baseline_bbox_first,
                "bbox_to_zm": baseline_zm_first,
                "complete_device_pipeline": baseline_descriptor_first,
            },
            "bbox_max_order": _summary(
                _samples(baseline_bbox, repeats, sample_count), 2
            ),
            "bbox_to_zm": _summary(
                _samples(baseline_zm, repeats, sample_count), 2
            ),
            "complete_device_pipeline": _summary(
                _samples(baseline_descriptor, repeats, sample_count), 2
            ),
        }
    }
    for frontier in PRECISION_FRONTIERS:
        def calculate_bbox(frontier=frontier):
            return calculate_bbox_moments_mixed_prototype(
                voxels,
                MAX_ORDER,
                radius[3],
                radius[4],
                radius[5],
                frontier,
            )

        bbox, bbox_first = _first_execution(calculate_bbox)

        def calculate_zm(frontier=frontier, bbox=bbox):
            return calculate_zm_mixed_prototype(
                bbox,
                MAX_ORDER,
                runtime.bbox_to_zm_cache.g_coefficients,
                runtime.bbox_to_zm_cache.pqr_indices,
                runtime.bbox_to_zm_cache.output_indices,
                runtime.bbox_to_zm_cache.clm,
                x64_cache.g_coefficients,
                x64_cache.clm,
                frontier,
            )

        def calculate_descriptor(frontier=frontier):
            return calculate_descriptor_from_voxels_mixed_prototype(
                voxels,
                max_order=MAX_ORDER,
                target_orders=TARGET_ORDERS,
                default_radius_multiplier=runtime.param[
                    "default_radius_multiplier"
                ],
                configured_bbox_cache=runtime.bbox_to_zm_cache,
                x64_bbox_cache=x64_cache,
                rotation_cache=runtime.rotation_cache,
                descriptor_cache=runtime.descriptor_cache,
                precision_frontier=frontier,
            )

        _, zm_first = _first_execution(calculate_zm)
        _, descriptor_first = _first_execution(calculate_descriptor)

        profiles[frontier] = {
            "first_execution_seconds": {
                "bbox_max_order": bbox_first,
                "bbox_to_zm": zm_first,
                "complete_device_pipeline": descriptor_first,
            },
            "bbox_max_order": _summary(
                _samples(calculate_bbox, repeats, sample_count), 2
            ),
            "bbox_to_zm": _summary(
                _samples(calculate_zm, repeats, sample_count), 2
            ),
            "complete_device_pipeline": _summary(
                _samples(calculate_descriptor, repeats, sample_count), 2
            ),
        }

    fastest = min(
        PRECISION_FRONTIERS,
        key=lambda frontier: profiles[frontier]["complete_device_pipeline"][
            "median_seconds"
        ],
    )
    payload = {
        "schema_version": 2,
        "backend": BACKEND,
        "configuration": {
            "max_order": MAX_ORDER,
            "target_orders": list(TARGET_ORDERS),
            "structures": ["6NT5", "6NT6"],
            "batch_size": 2,
            "samples": sample_count,
            "repeats_per_sample": repeats,
            "first_execution_note": (
                "Observed in one shared process; compiled subkernels may be reused "
                "by later profiles. Use warmed samples for candidate comparison."
            ),
        },
        "environment": {
            "python_platform": platform.platform(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "device": [str(device) for device in jax.devices()],
        },
        "profiles": profiles,
        "fastest_prototype": fastest,
        "production_accuracy_default": "moments_x64",
    }
    output_dir = Path(
        os.getenv(
            "ZMPY3D_MIXED_BENCHMARK_OUTPUT",
            str(Path(__file__).resolve().parent / "_simple_time_benchmark"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"mixed_precision_benchmark_{BACKEND}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
