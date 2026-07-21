"""CPU/GPU throughput harness for the production device-batched descriptor path."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

import ZMPY3D_JAX as z

BENCHMARK_BACKEND = os.getenv("ZMPY3D_BENCHMARK_BACKEND", "cpu").lower()
if BENCHMARK_BACKEND not in {"cpu", "gpu"}:
    raise ValueError("ZMPY3D_BENCHMARK_BACKEND must be 'cpu' or 'gpu'")
z.configure_for_scientific_computing(enable_x64=True, platform=BENCHMARK_BACKEND)

from ZMPY3D_JAX.ZMPY3D_CLI_BatchZM import (  # noqa: E402
    _BatchZMRuntime,
    _prepare_batch_runtime,
    _run_prepared_batch,
)
from ZMPY3D_JAX.lib.batched_descriptor import (  # noqa: E402
    calculate_descriptor_batch_from_voxels,
    calculate_descriptor_batch_staged,
    pad_voxel_batch,
)
from ZMPY3D_JAX.lib.fill_voxel_by_weight_density04 import (  # noqa: E402
    fill_voxel_by_weight_density_host,
)
from ZMPY3D_JAX.tests.utils.upstream_regression import REPO_ROOT, block_tree  # noqa: E402


BASE_STAGE_NAMES = (
    "bbox_order1",
    "radius_and_samples",
    "bbox_max_order",
    "bbox_to_zm",
    "descriptor_3dzd",
)
NORMALIZATION_ORDERS = tuple(range(2, 6))
NORMALIZATION_STAGE_PREFIXES = (
    "ab_candidates",
    "zm_rotation",
    "mean_invariant",
)
NORMALIZATION_REPRESENTATIONS = (
    "full_fixed",
    "analytic_compact",
    "analytic_compact_parity",
)
ROTATION_REDUCTIONS = ("scatter", "segmented_scan")
MOMENT_REDUCTIONS = ("scatter", "segmented_scan")


def _stage_names() -> tuple[str, ...]:
    return (
        *BASE_STAGE_NAMES,
        *(
            f"{prefix}_order_{order}"
            for order in NORMALIZATION_ORDERS
            for prefix in NORMALIZATION_STAGE_PREFIXES
        ),
        "descriptor_assembly",
    )


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _batch_sizes() -> tuple[int, ...]:
    configured = os.getenv("ZMPY3D_BATCH_SIZES", "1,4,16")
    try:
        values = tuple(
            dict.fromkeys(int(item.strip()) for item in configured.split(","))
        )
    except ValueError as error:
        raise ValueError(
            "ZMPY3D_BATCH_SIZES must be comma-separated positive integers"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise ValueError("ZMPY3D_BATCH_SIZES must be comma-separated positive integers")
    return values


def _time_call(function: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter_ns()
    result = function()
    block_tree(result)
    return result, (time.perf_counter_ns() - start) / 1e9


def _time_repeated(function: Callable[[], Any], repeats: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(repeats):
        block_tree(function())
    return (time.perf_counter_ns() - start) / repeats / 1e9


def _sample_pair(
    sequential: Callable[[], Any],
    batched: Callable[[], Any],
    *,
    repeats: int,
    sample_count: int,
) -> tuple[list[float], list[float]]:
    sequential_samples: list[float] = []
    batched_samples: list[float] = []
    for sample_index in range(sample_count):
        ordered = (
            ((sequential, sequential_samples), (batched, batched_samples))
            if sample_index % 2 == 0
            else ((batched, batched_samples), (sequential, sequential_samples))
        )
        for function, destination in ordered:
            destination.append(_time_repeated(function, repeats))
    return sequential_samples, batched_samples


def _sample_functions(
    functions: dict[str, Callable[[], Any]], *, repeats: int, sample_count: int
) -> dict[str, list[float]]:
    samples = {name: [] for name in functions}
    names = tuple(functions)
    for sample_index in range(sample_count):
        offset = sample_index % len(names)
        for name in (*names[offset:], *names[:offset]):
            samples[name].append(_time_repeated(functions[name], repeats))
    return samples


def _summary(values: list[float], protein_count: int) -> dict[str, Any]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.size == 0 or np.any(samples <= 0) or not np.all(np.isfinite(samples)):
        raise ValueError("timing samples must be positive, finite, and non-empty")
    median = float(np.median(samples))
    return {
        "samples_seconds": samples.tolist(),
        "median_seconds": median,
        "p25_seconds": float(np.percentile(samples, 25)),
        "p75_seconds": float(np.percentile(samples, 75)),
        "median_milliseconds_per_protein": 1000.0 * median / protein_count,
        "median_proteins_per_second": protein_count / median,
    }


def _comparison(
    sequential_samples: list[float], batched_samples: list[float], protein_count: int
) -> dict[str, Any]:
    sequential = _summary(sequential_samples, protein_count)
    batched = _summary(batched_samples, protein_count)
    sequential_median = float(sequential["median_seconds"])
    batched_median = float(batched["median_seconds"])
    return {
        "sequential": sequential,
        "batched": batched,
        "sequential_over_batched_time_ratio": sequential_median / batched_median,
        "batched_over_sequential_time_ratio": batched_median / sequential_median,
    }


def _stage_categories() -> dict[str, tuple[str, ...]]:
    return {
        "moments": (
            "bbox_order1",
            "radius_and_samples",
            "bbox_max_order",
            "bbox_to_zm",
        ),
        "3dzd": ("descriptor_3dzd",),
        "ab_candidates": tuple(
            f"ab_candidates_order_{order}" for order in NORMALIZATION_ORDERS
        ),
        "zm_rotation": tuple(
            f"zm_rotation_order_{order}" for order in NORMALIZATION_ORDERS
        ),
        "mean_invariant": tuple(
            f"mean_invariant_order_{order}" for order in NORMALIZATION_ORDERS
        ),
        "assembly": ("descriptor_assembly",),
    }


def _aggregate_stage_samples(
    stage_samples: dict[str, list[float]], stage_names: tuple[str, ...]
) -> list[float]:
    return [
        sum(stage_samples[name][sample_index] for name in stage_names)
        for sample_index in range(len(next(iter(stage_samples.values()))))
    ]


def _rank_stages(stage_summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "stage": name,
            "median_seconds": float(summary["median_seconds"]),
            "median_milliseconds_per_protein": float(
                summary["median_milliseconds_per_protein"]
            ),
        }
        for name, summary in stage_summaries.items()
    ]
    rows.sort(key=lambda row: float(row["median_seconds"]), reverse=True)
    median_total = sum(float(row["median_seconds"]) for row in rows)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["synchronized_time_percent"] = (
            100.0 * float(row["median_seconds"]) / median_total
        )
    return rows


def _host_voxels(paths: list[str], runtime: _BatchZMRuntime) -> list[np.ndarray]:
    voxels = []
    for path in paths:
        xyz, residues = z.get_pdb_xyz_ca(path)
        voxel, _ = fill_voxel_by_weight_density_host(
            xyz,
            residues,
            runtime.param["residue_weight_map"],
            1.0,
            runtime.residue_box[1.0],
        )
        voxels.append(voxel)
    return voxels


def _run_core(
    voxels: jax.Array,
    runtime: _BatchZMRuntime,
    mode: int = 2,
    normalization_representation: str = "analytic_compact",
    rotation_reduction: str = "auto",
    moment_reduction: str = "auto",
) -> z.DescriptorVector:
    return calculate_descriptor_batch_from_voxels(
        voxels,
        max_order=6,
        max_target_order=5,
        mode=mode,
        default_radius_multiplier=runtime.param["default_radius_multiplier"],
        bbox_to_zm_cache=runtime.bbox_to_zm_cache,
        rotation_cache=runtime.rotation_cache,
        descriptor_cache=runtime.descriptor_cache,
        normalization_representation=normalization_representation,
        rotation_reduction=rotation_reduction,
        moment_reduction=moment_reduction,
    )


def _run_sequential_core(
    device_voxels: list[jax.Array], runtime: _BatchZMRuntime
) -> z.DescriptorVector:
    results = [_run_core(voxel, runtime) for voxel in device_voxels]
    return z.DescriptorVector(
        values=jnp.concatenate([item.values for item in results], axis=0),
        is_valid=jnp.concatenate([item.is_valid for item in results], axis=0),
    )


def _run_staged_core(
    voxels: jax.Array,
    runtime: _BatchZMRuntime,
    stage_executor=None,
):
    return calculate_descriptor_batch_staged(
        voxels,
        max_order=6,
        max_target_order=5,
        mode=2,
        default_radius_multiplier=runtime.param["default_radius_multiplier"],
        bbox_to_zm_cache=runtime.bbox_to_zm_cache,
        rotation_cache=runtime.rotation_cache,
        descriptor_cache=runtime.descriptor_cache,
        stage_executor=stage_executor,
    )


def _profile_staged_batch(voxels: jax.Array, runtime: _BatchZMRuntime, repeats: int):
    totals_ns = {name: 0 for name in _stage_names()}
    last_result = None
    last_candidates = None
    for _ in range(repeats):
        observed: set[str] = set()

        def execute(name: str, function: Callable[[], Any]) -> Any:
            assert name in totals_ns
            assert name not in observed
            observed.add(name)
            start = time.perf_counter_ns()
            result = function()
            block_tree(result)
            totals_ns[name] += time.perf_counter_ns() - start
            return result

        last_result, last_candidates = _run_staged_core(voxels, runtime, execute)
        assert observed == set(totals_ns)

    return (
        last_result,
        last_candidates,
        {name: totals_ns[name] / repeats / 1e9 for name in totals_ns},
    )


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _output_path() -> Path:
    configured = os.getenv("ZMPY3D_BATCH_BENCHMARK_OUTPUT") or os.getenv(
        "ZMPY3D_BENCHMARK_OUTPUT"
    )
    output_dir = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parent / "_simple_time_benchmark"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return (
        output_dir / f"batched_pipeline_benchmark_{BENCHMARK_BACKEND}_{timestamp}.json"
    )


def _summarize_stage_profile(
    stage_samples: dict[str, list[float]],
    candidates_by_order: dict[int, Any],
    protein_count: int,
) -> dict[str, Any]:
    stage_summaries = {
        name: _summary(stage_samples[name], protein_count) for name in _stage_names()
    }
    categories = {
        name: _summary(_aggregate_stage_samples(stage_samples, members), protein_count)
        for name, members in _stage_categories().items()
    }
    normalization_by_order = {}
    for order in NORMALIZATION_ORDERS:
        candidates = candidates_by_order[order]
        normalization_by_order[str(order)] = {
            "fixed_slots_per_protein": int(candidates.pairs.shape[1]),
            "valid_slots_per_protein": np.asarray(
                jnp.sum(candidates.is_valid, axis=1)
            ).tolist(),
            "stages": {
                prefix: stage_summaries[f"{prefix}_order_{order}"]
                for prefix in NORMALIZATION_STAGE_PREFIXES
            },
        }
    ranking = _rank_stages(stage_summaries)
    return {
        "synchronization": "block_after_each_stage",
        "note": (
            "Synchronized stage timings are diagnostic and do not reconstruct the fused "
            "device-core latency."
        ),
        "stages": stage_summaries,
        "categories": categories,
        "normalization_by_order": normalization_by_order,
        "ranking": ranking,
        "bottleneck": ranking[0],
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    assert payload["schema_version"] == 5
    assert payload["configuration"]["batch_sizes"]
    for workload in payload["results"]["workloads"].values():
        assert 0 < workload["padding_utilization"] <= 1
        for section in ("transfer", "device_core", "prepared_end_to_end"):
            comparison = workload[section]
            assert comparison["sequential_over_batched_time_ratio"] > 0
            np.testing.assert_allclose(
                comparison["sequential_over_batched_time_ratio"]
                * comparison["batched_over_sequential_time_ratio"],
                1.0,
            )
        assert set(workload["mode_profile"]) == {
            "mode_0_normalization",
            "mode_1_3dzd",
            "mode_2_combined",
        }
        assert set(workload["normalization_representation_profile"]) == set(
            NORMALIZATION_REPRESENTATIONS
        )
        assert set(workload["rotation_reduction_profile"]) == set(
            ROTATION_REDUCTIONS
        )
        assert set(workload["moment_reduction_profile"]) == set(
            MOMENT_REDUCTIONS
        )
        stage_profile = workload["stage_profile"]
        assert set(stage_profile["stages"]) == set(_stage_names())
        assert set(stage_profile["categories"]) == set(_stage_categories())
        assert len(stage_profile["ranking"]) == len(_stage_names())
        assert stage_profile["bottleneck"] == stage_profile["ranking"][0]
        np.testing.assert_allclose(
            sum(row["synchronized_time_percent"] for row in stage_profile["ranking"]),
            100.0,
        )
        for order in NORMALIZATION_ORDERS:
            metadata = stage_profile["normalization_by_order"][str(order)]
            expected_fixed = 16 if order % 2 == 0 else 8
            expected_valid = 8 if order % 2 == 0 else 4
            assert metadata["fixed_slots_per_protein"] == expected_fixed
            assert len(metadata["valid_slots_per_protein"]) == int(
                workload["padded_voxel_shape"][0]
            )
            assert set(metadata["valid_slots_per_protein"]) == {expected_valid}


def test_summary_and_comparison_account_per_protein() -> None:
    summary = _summary([2.0, 4.0, 6.0], protein_count=2)
    assert summary["median_seconds"] == 4.0
    assert summary["median_milliseconds_per_protein"] == 2000.0
    assert summary["median_proteins_per_second"] == 0.5
    comparison = _comparison([4.0], [2.0], protein_count=2)
    assert comparison["sequential_over_batched_time_ratio"] == 2.0
    assert comparison["batched_over_sequential_time_ratio"] == 0.5


def test_stage_aggregation_and_ranking() -> None:
    samples = {name: [1.0, 2.0] for name in _stage_names()}
    assert _aggregate_stage_samples(samples, ("bbox_order1", "bbox_to_zm")) == [
        2.0,
        4.0,
    ]
    summaries = {
        name: {
            "median_seconds": float(index + 1),
            "median_milliseconds_per_protein": 1.0,
        }
        for index, name in enumerate(_stage_names())
    }
    ranking = _rank_stages(summaries)
    assert ranking[0]["stage"] == _stage_names()[-1]
    assert ranking[0]["rank"] == 1
    np.testing.assert_allclose(
        sum(row["synchronized_time_percent"] for row in ranking), 100.0
    )


def test_batched_pipeline_throughput_snapshot() -> None:
    assert jax.default_backend() == BENCHMARK_BACKEND
    repeats = _positive_env_int("ZMPY3D_BATCH_REPEATS", 2)
    sample_count = _positive_env_int("ZMPY3D_BATCH_SAMPLES", 5)
    batch_sizes = _batch_sizes()
    fixture_paths = [str(REPO_ROOT / "6NT5.pdb"), str(REPO_ROOT / "6NT6.pdb")]

    runtime, setup_seconds = _time_call(lambda: _prepare_batch_runtime(1.0, 6))
    workloads = {}
    for batch_size in batch_sizes:
        paths = [fixture_paths[index % 2] for index in range(batch_size)]
        host_voxels, host_prepare_once = _time_call(
            lambda: _host_voxels(paths, runtime)
        )
        padded_host = pad_voxel_batch(host_voxels)

        def transfer_sequential():
            return [
                jnp.asarray(voxel[None, ...], dtype=z.FLOAT_DTYPE)
                for voxel in host_voxels
            ]

        def transfer_batched():
            return jnp.asarray(padded_host, dtype=z.FLOAT_DTYPE)

        sequential_device = transfer_sequential()
        batched_device = transfer_batched()
        block_tree((sequential_device, batched_device))

        def sequential_core():
            return _run_sequential_core(sequential_device, runtime)

        def batched_core():
            return _run_core(batched_device, runtime)

        sequential_result = sequential_core()
        batched_result = batched_core()
        block_tree((sequential_result, batched_result))
        np.testing.assert_allclose(
            np.asarray(batched_result.values),
            np.asarray(sequential_result.values),
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            np.asarray(batched_result.is_valid), np.asarray(sequential_result.is_valid)
        )

        jax.clear_caches()
        _, sequential_first = _time_call(sequential_core)
        jax.clear_caches()
        _, batched_first = _time_call(batched_core)
        sequential_core()
        batched_core()

        transfer_samples = _sample_pair(
            transfer_sequential,
            transfer_batched,
            repeats=repeats,
            sample_count=sample_count,
        )
        core_samples = _sample_pair(
            sequential_core,
            batched_core,
            repeats=repeats,
            sample_count=sample_count,
        )

        mode_functions = {
            "mode_0_normalization": lambda: _run_core(batched_device, runtime, mode=0),
            "mode_1_3dzd": lambda: _run_core(batched_device, runtime, mode=1),
            "mode_2_combined": lambda: _run_core(batched_device, runtime, mode=2),
        }
        mode_results = {name: function() for name, function in mode_functions.items()}
        block_tree(mode_results)
        np.testing.assert_allclose(
            np.asarray(mode_results["mode_2_combined"].values),
            np.concatenate(
                (
                    np.asarray(mode_results["mode_1_3dzd"].values),
                    np.asarray(mode_results["mode_0_normalization"].values),
                ),
                axis=1,
            ),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            np.asarray(mode_results["mode_2_combined"].is_valid),
            np.concatenate(
                (
                    np.asarray(mode_results["mode_1_3dzd"].is_valid),
                    np.asarray(mode_results["mode_0_normalization"].is_valid),
                ),
                axis=1,
            ),
        )
        mode_samples = _sample_functions(
            mode_functions, repeats=repeats, sample_count=sample_count
        )

        representation_functions = {
            representation: (
                lambda representation=representation: _run_core(
                    batched_device,
                    runtime,
                    mode=0,
                    normalization_representation=representation,
                )
            )
            for representation in NORMALIZATION_REPRESENTATIONS
        }
        representation_results = {
            name: function() for name, function in representation_functions.items()
        }
        block_tree(representation_results)
        for representation in NORMALIZATION_REPRESENTATIONS[1:]:
            np.testing.assert_allclose(
                np.asarray(representation_results[representation].values),
                np.asarray(representation_results["full_fixed"].values),
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
            )
            np.testing.assert_array_equal(
                np.asarray(representation_results[representation].is_valid),
                np.asarray(representation_results["full_fixed"].is_valid),
            )
        representation_samples = _sample_functions(
            representation_functions, repeats=repeats, sample_count=sample_count
        )
        reduction_functions = {
            reduction: (
                lambda reduction=reduction: _run_core(
                    batched_device,
                    runtime,
                    mode=0,
                    rotation_reduction=reduction,
                )
            )
            for reduction in ROTATION_REDUCTIONS
        }
        reduction_results = {
            name: function() for name, function in reduction_functions.items()
        }
        block_tree(reduction_results)
        np.testing.assert_allclose(
            np.asarray(reduction_results["segmented_scan"].values),
            np.asarray(reduction_results["scatter"].values),
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            np.asarray(reduction_results["segmented_scan"].is_valid),
            np.asarray(reduction_results["scatter"].is_valid),
        )
        reduction_samples = _sample_functions(
            reduction_functions, repeats=repeats, sample_count=sample_count
        )
        moment_reduction_functions = {
            reduction: (
                lambda reduction=reduction: _run_core(
                    batched_device,
                    runtime,
                    mode=1,
                    moment_reduction=reduction,
                )
            )
            for reduction in MOMENT_REDUCTIONS
        }
        moment_reduction_results = {
            name: function()
            for name, function in moment_reduction_functions.items()
        }
        block_tree(moment_reduction_results)
        np.testing.assert_allclose(
            np.asarray(moment_reduction_results["segmented_scan"].values),
            np.asarray(moment_reduction_results["scatter"].values),
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            np.asarray(moment_reduction_results["segmented_scan"].is_valid),
            np.asarray(moment_reduction_results["scatter"].is_valid),
        )
        moment_reduction_samples = _sample_functions(
            moment_reduction_functions,
            repeats=repeats,
            sample_count=sample_count,
        )

        staged_result, candidates_by_order = _run_staged_core(batched_device, runtime)
        block_tree((staged_result, candidates_by_order))
        np.testing.assert_allclose(
            np.asarray(staged_result.values),
            np.asarray(batched_result.values),
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            np.asarray(staged_result.is_valid), np.asarray(batched_result.is_valid)
        )
        stage_samples = {name: [] for name in _stage_names()}
        for _ in range(sample_count):
            _, candidates_by_order, profile = _profile_staged_batch(
                batched_device, runtime, repeats
            )
            for name in _stage_names():
                stage_samples[name].append(profile[name])

        def sequential_end_to_end():
            return _run_prepared_batch(
                paths,
                grid_width=1.0,
                max_order=6,
                max_target_order=5,
                mode=2,
                batch_size=1,
                runtime=runtime,
            )

        def batched_end_to_end():
            return _run_prepared_batch(
                paths,
                grid_width=1.0,
                max_order=6,
                max_target_order=5,
                mode=2,
                batch_size=batch_size,
                runtime=runtime,
            )

        sequential_end_to_end()
        batched_end_to_end()
        end_to_end_samples = _sample_pair(
            sequential_end_to_end,
            batched_end_to_end,
            repeats=repeats,
            sample_count=sample_count,
        )

        padded_cells = int(np.prod(padded_host.shape))
        native_cells = sum(int(np.prod(voxel.shape)) for voxel in host_voxels)
        workloads[str(batch_size)] = {
            "input_sequence": [Path(path).stem for path in paths],
            "native_voxel_shapes": [list(voxel.shape) for voxel in host_voxels],
            "padded_voxel_shape": list(padded_host.shape),
            "padding_utilization": native_cells / padded_cells,
            "host_preparation_once_seconds": host_prepare_once,
            "first_device_execution": {
                "sequential_seconds": sequential_first,
                "batched_seconds": batched_first,
            },
            "transfer": _comparison(*transfer_samples, protein_count=batch_size),
            "device_core": _comparison(*core_samples, protein_count=batch_size),
            "mode_profile": {
                name: _summary(samples, batch_size)
                for name, samples in mode_samples.items()
            },
            "normalization_representation_profile": {
                name: _summary(samples, batch_size)
                for name, samples in representation_samples.items()
            },
            "rotation_reduction_profile": {
                name: _summary(samples, batch_size)
                for name, samples in reduction_samples.items()
            },
            "moment_reduction_profile": {
                name: _summary(samples, batch_size)
                for name, samples in moment_reduction_samples.items()
            },
            "stage_profile": _summarize_stage_profile(
                stage_samples, candidates_by_order, batch_size
            ),
            "prepared_end_to_end": _comparison(
                *end_to_end_samples, protein_count=batch_size
            ),
        }

    payload = {
        "schema_version": 5,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_mode": "informational",
        "configuration": {
            "batch_sizes": list(batch_sizes),
            "inputs": ["6NT5.pdb", "6NT6.pdb"],
            "grid_width": 1.0,
            "max_order": 6,
            "max_target_order": 5,
            "mode": 2,
            "samples": sample_count,
            "repeats_per_sample": repeats,
            "jax_x64_enabled": bool(jax.config.x64_enabled),
            "jax_requested_backend": BENCHMARK_BACKEND,
            "voxelization": "sequential_numpy_host",
            "padding": "input_order_chunk_high_side_zero_padding",
            "stage_profile": "device_resident_block_after_each_stage",
            "profiled_modes": {
                "0": "normalization_only",
                "1": "3dzd_only",
                "2": "combined",
            },
            "normalization_representations": list(
                NORMALIZATION_REPRESENTATIONS
            ),
            "rotation_reductions": list(ROTATION_REDUCTIONS),
            "moment_reductions": list(MOMENT_REDUCTIONS),
        },
        "environment": {
            "git_revision": _git_revision(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "results": {"runtime_setup_seconds": setup_seconds, "workloads": workloads},
    }
    _validate_payload(payload)
    _output_path().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
