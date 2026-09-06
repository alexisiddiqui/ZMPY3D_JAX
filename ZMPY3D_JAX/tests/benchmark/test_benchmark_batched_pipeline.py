"""CPU/GPU throughput harness for the production device-batched descriptor path."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import pytest

import ZMPY3D_JAX as z

BENCHMARK_BACKEND = os.getenv("ZMPY3D_BENCHMARK_BACKEND", "cpu").lower()
if BENCHMARK_BACKEND not in {"cpu", "gpu"}:
    raise ValueError("ZMPY3D_BENCHMARK_BACKEND must be 'cpu' or 'gpu'")
z.configure_for_scientific_computing(enable_x64=False, platform=BENCHMARK_BACKEND)

from ZMPY3D_JAX.ZMPY3D_CLI_BatchZM import (  # noqa: E402
    _BatchZMRuntime,
    _prepare_batch_runtime,
    _prepare_descriptor_runner,
    _run_prepared_batch,
)
from ZMPY3D_JAX.lib.batched_descriptor import (  # noqa: E402
    _calculate_ab_compact_candidate_group_batch,
    _calculate_ab_compact_candidates_batch,
    _calculate_normalized_mean_compact_batch,
    _calculate_rotation_batch,
    _calculate_rotation_flat_batch,
    calculate_descriptor_batch_from_voxels,
    calculate_descriptor_batch_staged,
    pad_voxel_batch,
)
from ZMPY3D_JAX.lib.fill_voxel_by_weight_density04 import (  # noqa: E402
    fill_voxel_by_weight_density_host,
)
from ZMPY3D_JAX.tests.utils.upstream_regression import REPO_ROOT, block_tree  # noqa: E402


MAX_ORDER = int(os.getenv("ZMPY3D_BATCH_MAX_ORDER", "20"))
if MAX_ORDER not in (6, 20, 40):
    raise ValueError("ZMPY3D_BATCH_MAX_ORDER must be 6, 20, or 40")
MAX_TARGET_ORDER = min(5, MAX_ORDER)

BASE_STAGE_NAMES = (
    "bbox_order1",
    "radius_and_samples",
    "bbox_max_order",
    "bbox_to_zm",
    "descriptor_3dzd",
)
NORMALIZATION_ORDERS = tuple(range(2, MAX_TARGET_ORDER + 1))
NORMALIZATION_STAGE_PREFIXES = (
    "ab_candidates",
    "zm_rotation",
    "mean_invariant",
)
NORMALIZATION_REPRESENTATIONS = (
    "full_fixed",
    "companion_compact",
    "companion_compact_grouped",
    "companion_compact_grouped_flat",
    "analytic_compact",
)
ROTATION_REDUCTIONS = ("scatter", "segmented_scan")
MOMENT_REDUCTIONS = ("production_auto",)


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
    configured = os.getenv("ZMPY3D_BATCH_SIZES", "2,16")
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


def _benchmark_input_paths() -> list[str]:
    configured = os.getenv("ZMPY3D_BATCH_INPUT_MANIFEST")
    if not configured:
        return [str(REPO_ROOT / "6NT5.pdb"), str(REPO_ROOT / "6NT6.pdb")]

    manifest = Path(configured).expanduser()
    if not manifest.is_file():
        raise ValueError(f"ZMPY3D_BATCH_INPUT_MANIFEST is not a file: {manifest}")
    paths = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line).expanduser()
        paths.append(str(path if path.is_absolute() else manifest.parent / path))
    if not paths:
        raise ValueError(f"ZMPY3D_BATCH_INPUT_MANIFEST is empty: {manifest}")
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise ValueError(
            "ZMPY3D_BATCH_INPUT_MANIFEST contains missing PDB files: "
            + ", ".join(missing)
        )
    return paths


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


def _profile_host_preparation(
    paths: list[str], runtime: _BatchZMRuntime, grid_width: float = 1.0
) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    """Run and time the production parse -> fill -> pad host sequence once."""
    total_start = time.perf_counter_ns()
    host_voxels = []
    structures = []
    parse_seconds = 0.0
    accumulate_seconds = 0.0
    residue_box = runtime.residue_box[grid_width]

    for path in paths:
        parse_start = time.perf_counter_ns()
        xyz, residues = z.get_pdb_xyz_ca(path)
        parse_elapsed = (time.perf_counter_ns() - parse_start) / 1e9

        accumulate_start = time.perf_counter_ns()
        voxel, _corner = fill_voxel_by_weight_density_host(
            xyz,
            residues,
            runtime.param["residue_weight_map"],
            grid_width,
            residue_box,
        )
        accumulate_elapsed = (time.perf_counter_ns() - accumulate_start) / 1e9

        parse_seconds += parse_elapsed
        accumulate_seconds += accumulate_elapsed
        host_voxels.append(voxel)
        structures.append(
            {
                "input": Path(path).stem,
                "ca_count": int(np.asarray(xyz).shape[0]),
                "voxel_shape": list(voxel.shape),
                "native_cells": int(np.prod(voxel.shape)),
                "parse_seconds": parse_elapsed,
                "accumulate_seconds": accumulate_elapsed,
            }
        )

    pad_start = time.perf_counter_ns()
    padded_host = pad_voxel_batch(host_voxels)
    pad_seconds = (time.perf_counter_ns() - pad_start) / 1e9
    total_seconds = (time.perf_counter_ns() - total_start) / 1e9
    return host_voxels, padded_host, {
        "total_seconds": total_seconds,
        "parse_seconds": parse_seconds,
        "accumulate_seconds": accumulate_seconds,
        "pad_seconds": pad_seconds,
        "structures": structures,
    }


def _sample_host_preparation(
    paths: list[str],
    runtime: _BatchZMRuntime,
    *,
    grid_width: float,
    repeats: int,
    sample_count: int,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any], dict[str, list[float]]]:
    stage_samples = {
        "parse": [],
        "accumulate": [],
        "pad": [],
        "total": [],
    }
    first_profile = None
    host_voxels = None
    padded_host = None
    for _sample_index in range(sample_count):
        for _repeat_index in range(repeats):
            host_voxels, padded_host, profile = _profile_host_preparation(
                paths, runtime, grid_width
            )
            if first_profile is None:
                first_profile = profile
            stage_samples["parse"].append(profile["parse_seconds"])
            stage_samples["accumulate"].append(profile["accumulate_seconds"])
            stage_samples["pad"].append(profile["pad_seconds"])
            stage_samples["total"].append(profile["total_seconds"])
    assert host_voxels is not None
    assert padded_host is not None
    assert first_profile is not None
    return host_voxels, padded_host, first_profile, stage_samples


def _run_core(
    voxels: jax.Array,
    runtime: _BatchZMRuntime,
    mode: int = 2,
    normalization_representation: str = "companion_compact_grouped",
    rotation_reduction: str = "auto",
    moment_reduction: str = "auto",
) -> z.DescriptorVector:
    return calculate_descriptor_batch_from_voxels(
        voxels,
        max_order=MAX_ORDER,
        max_target_order=MAX_TARGET_ORDER,
        mode=mode,
        default_radius_multiplier=runtime.param["default_radius_multiplier"],
        bbox_to_zm_cache=runtime.bbox_to_zm_cache,
        x64_bbox_to_zm_cache=runtime.x64_bbox_to_zm_cache,
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
        max_order=MAX_ORDER,
        max_target_order=MAX_TARGET_ORDER,
        mode=2,
        default_radius_multiplier=runtime.param["default_radius_multiplier"],
        bbox_to_zm_cache=runtime.bbox_to_zm_cache,
        x64_bbox_to_zm_cache=runtime.x64_bbox_to_zm_cache,
        rotation_cache=runtime.rotation_cache,
        descriptor_cache=runtime.descriptor_cache,
        stage_executor=stage_executor,
        normalization_representation="companion_compact",
    )


def _profile_staged_batch(voxels: jax.Array, runtime: _BatchZMRuntime, repeats: int):
    totals_ns = {name: 0 for name in _stage_names()}
    last_result = None
    last_candidates = None
    last_outputs = None
    for _ in range(repeats):
        observed: set[str] = set()
        outputs: dict[str, Any] = {}

        def execute(name: str, function: Callable[[], Any]) -> Any:
            assert name in totals_ns
            assert name not in observed
            observed.add(name)
            start = time.perf_counter_ns()
            result = function()
            block_tree(result)
            totals_ns[name] += time.perf_counter_ns() - start
            outputs[name] = result
            return result

        last_result, last_candidates = _run_staged_core(voxels, runtime, execute)
        last_outputs = outputs
        assert observed == set(totals_ns)

    return (
        last_result,
        last_candidates,
        last_outputs,
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
    return output_dir / f"batched_pipeline_profile_{BENCHMARK_BACKEND}.json"


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
    assert payload["schema_version"] == 8
    assert payload["configuration"]["batch_sizes"]
    for workload in payload["results"]["workloads"].values():
        assert 0 < workload["padding_utilization"] <= 1
        host_preparation = workload["host_preparation"]
        assert set(host_preparation["once_breakdown_seconds"]) == {
            "parse",
            "accumulate",
            "pad",
        }
        assert set(host_preparation["warmed"]) == {
            "parse",
            "accumulate",
            "pad",
            "total",
        }
        assert host_preparation["once_seconds"] > 0
        assert sum(host_preparation["once_breakdown_seconds"].values()) <= (
            host_preparation["once_seconds"] * 1.05
        )
        for summary in host_preparation["warmed"].values():
            assert summary["median_seconds"] > 0

        storage = workload["voxel_storage"]
        assert storage["native_cells"] > 0
        assert storage["padded_cells"] >= storage["native_cells"]
        assert storage["native_bytes"] > 0
        assert storage["padded_bytes"] >= storage["native_bytes"]
        assert storage["padding_fraction"] == pytest.approx(
            1.0 - workload["padding_utilization"]
        )
        transfer_batched = workload["transfer_batched"]
        assert transfer_batched["bytes"] == storage["padded_bytes"]
        assert transfer_batched["bytes_per_protein"] == pytest.approx(
            storage["padded_bytes"] / len(workload["input_sequence"])
        )
        assert transfer_batched["first_seconds"] > 0
        assert transfer_batched["warmed"]["median_seconds"] > 0
        checkpoint_gate = workload["checkpoint_0_gate"]
        assert checkpoint_gate["evaluated"] in (True, False)
        if checkpoint_gate["evaluated"]:
            assert checkpoint_gate["passed"] in (True, False)
        else:
            assert checkpoint_gate["passed"] is None
        assert checkpoint_gate["accumulate_plus_transfer_milliseconds_per_protein"] > 0
        assert 0 < checkpoint_gate["accumulate_plus_transfer_share"] <= 1
        for section in (
            "transfer",
            "device_core",
            "compiled_device_core",
            "prepared_end_to_end",
        ):
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
            expected_fixed = 8 if order % 2 == 0 else 4
            expected_valid = expected_fixed
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
    fixture_paths = _benchmark_input_paths()

    runtime, setup_seconds = _time_call(
        lambda: _prepare_batch_runtime(1.0, MAX_ORDER)
    )
    workloads = {}
    for batch_size in batch_sizes:
        paths = [fixture_paths[index % len(fixture_paths)] for index in range(batch_size)]
        (
            host_voxels,
            padded_host,
            host_prepare_profile,
            host_prepare_samples,
        ) = _sample_host_preparation(
            paths,
            runtime,
            grid_width=1.0,
            repeats=repeats,
            sample_count=sample_count,
        )

        def transfer_sequential():
            return [
                jnp.asarray(voxel[None, ...], dtype=z.FLOAT_DTYPE)
                for voxel in host_voxels
            ]

        def transfer_batched():
            return jnp.asarray(padded_host, dtype=z.FLOAT_DTYPE)

        sequential_device, _sequential_transfer_first = _time_call(transfer_sequential)
        batched_device, batched_transfer_first = _time_call(transfer_batched)
        block_tree((sequential_device, batched_device))

        def sequential_core():
            return _run_sequential_core(sequential_device, runtime)

        def batched_core():
            return _run_core(batched_device, runtime)

        compiled_core = jax.jit(partial(_run_core, runtime=runtime))

        sequential_result = sequential_core()
        batched_result = batched_core()
        block_tree((sequential_result, batched_result))
        np.testing.assert_allclose(
            np.asarray(batched_result.values),
            np.asarray(sequential_result.values),
            rtol=2e-3,
            atol=2e-3,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            np.asarray(batched_result.is_valid), np.asarray(sequential_result.is_valid)
        )

        jax.clear_caches()
        _, sequential_first = _time_call(sequential_core)
        jax.clear_caches()
        _, batched_first = _time_call(batched_core)
        _, compiled_first = _time_call(lambda: compiled_core(batched_device))
        sequential_core()
        batched_core()
        compiled_result = compiled_core(batched_device)
        block_tree(compiled_result)
        np.testing.assert_allclose(
            np.asarray(compiled_result.values),
            np.asarray(batched_result.values),
            rtol=2e-5,
            atol=5e-6,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            np.asarray(compiled_result.is_valid), np.asarray(batched_result.is_valid)
        )

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
        compiled_core_samples = _sample_pair(
            batched_core,
            lambda: compiled_core(batched_device),
            repeats=repeats,
            sample_count=sample_count,
        )
        trace_root = os.getenv("ZMPY3D_BATCH_TRACE_DIR")
        if trace_root and batch_size == max(batch_sizes):
            trace_dir = Path(trace_root).expanduser() / (
                f"{BENCHMARK_BACKEND}_order{MAX_ORDER}_batch{batch_size}"
            )
            trace_dir.mkdir(parents=True, exist_ok=True)
            with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
                with jax.profiler.TraceAnnotation("separate_device_pipeline"):
                    block_tree(batched_core())
                with jax.profiler.TraceAnnotation("whole_pipeline_jit"):
                    block_tree(compiled_core(batched_device))

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
                rtol=2e-3,
                atol=2e-3,
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
            rtol=2e-3,
            atol=2e-3,
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
            "production_auto": lambda: _run_core(
                batched_device, runtime, mode=1
            )
        }
        moment_reduction_results = {
            name: function()
            for name, function in moment_reduction_functions.items()
        }
        block_tree(moment_reduction_results)
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
        stage_outputs = None
        for _ in range(sample_count):
            _, candidates_by_order, stage_outputs, profile = _profile_staged_batch(
                batched_device, runtime, repeats
            )
            for name in _stage_names():
                stage_samples[name].append(profile[name])

        assert stage_outputs is not None
        raw = stage_outputs["bbox_to_zm"][1]
        rotation = runtime.rotation_cache
        normalization_functions = {
            f"order_{order}": (
                lambda order=order: _calculate_normalized_mean_compact_batch(
                    raw,
                    order,
                    rotation.binomial,
                    rotation.max_order,
                    rotation.clm,
                    rotation.s_id,
                    rotation.n,
                    rotation.l,
                    rotation.m,
                    rotation.mu,
                    rotation.k,
                    rotation.is_nlm_value,
                    "auto",
                )
            )
            for order in NORMALIZATION_ORDERS
        }
        normalization_method_samples = _sample_functions(
            normalization_functions, repeats=repeats, sample_count=sample_count
        )
        candidate_solver_functions = {
            "odd_companion_order_3": lambda: _calculate_ab_compact_candidates_batch(
                raw, 3, "companion"
            ),
            "odd_analytic_order_3": lambda: _calculate_ab_compact_candidates_batch(
                raw, 3, "analytic_odd"
            ),
            "odd_companion_order_5": lambda: _calculate_ab_compact_candidates_batch(
                raw, 5, "companion"
            ),
            "odd_analytic_order_5": lambda: _calculate_ab_compact_candidates_batch(
                raw, 5, "analytic_odd"
            ),
            "even_companion_separate": lambda: (
                _calculate_ab_compact_candidates_batch(raw, 2, "companion"),
                _calculate_ab_compact_candidates_batch(raw, 4, "companion"),
            ),
            "even_companion_grouped": lambda: (
                _calculate_ab_compact_candidate_group_batch(
                    raw, (2, 4), "companion"
                )
            ),
        }
        candidate_solver_samples = _sample_functions(
            candidate_solver_functions, repeats=repeats, sample_count=sample_count
        )
        rotation_candidates = _calculate_ab_compact_candidates_batch(
            raw, 2, "companion"
        )
        rotation_arguments = (
            raw,
            rotation_candidates.pairs,
            rotation.max_order,
            rotation.binomial,
            rotation.clm,
            rotation.s_id,
            rotation.n,
            rotation.l,
            rotation.m,
            rotation.mu,
            rotation.k,
            rotation.is_nlm_value,
            "auto",
        )
        rotation_layout_functions = {
            "nested": lambda: _calculate_rotation_batch(*rotation_arguments),
            "flattened": lambda: _calculate_rotation_flat_batch(*rotation_arguments),
        }
        rotation_layout_results = {
            name: function() for name, function in rotation_layout_functions.items()
        }
        block_tree(rotation_layout_results)
        np.testing.assert_array_equal(
            np.asarray(rotation_layout_results["flattened"]),
            np.asarray(rotation_layout_results["nested"]),
        )
        rotation_layout_samples = _sample_functions(
            rotation_layout_functions,
            repeats=repeats,
            sample_count=sample_count,
        )
        odd_normalization_functions = {}
        for order in (3, 5):
            for strategy in ("companion", "analytic_odd"):
                odd_normalization_functions[f"order_{order}_{strategy}"] = (
                    lambda order=order, strategy=strategy: (
                        _calculate_normalized_mean_compact_batch(
                            raw,
                            order,
                            rotation.binomial,
                            rotation.max_order,
                            rotation.clm,
                            rotation.s_id,
                            rotation.n,
                            rotation.l,
                            rotation.m,
                            rotation.mu,
                            rotation.k,
                            rotation.is_nlm_value,
                            "auto",
                            strategy,
                        )
                    )
                )
        odd_normalization_results = {
            name: function()
            for name, function in odd_normalization_functions.items()
        }
        block_tree(odd_normalization_results)
        for order in (3, 5):
            np.testing.assert_allclose(
                np.asarray(
                    odd_normalization_results[f"order_{order}_analytic_odd"]
                ),
                np.asarray(odd_normalization_results[f"order_{order}_companion"]),
                rtol=2e-3,
                atol=2e-3,
                equal_nan=True,
            )
        odd_normalization_samples = _sample_functions(
            odd_normalization_functions,
            repeats=repeats,
            sample_count=sample_count,
        )

        def sequential_end_to_end():
            return _run_prepared_batch(
                paths,
                grid_width=1.0,
                max_order=MAX_ORDER,
                max_target_order=MAX_TARGET_ORDER,
                mode=2,
                batch_size=1,
                runtime=runtime,
                descriptor_runner=prepared_runner,
            )

        def batched_end_to_end():
            return _run_prepared_batch(
                paths,
                grid_width=1.0,
                max_order=MAX_ORDER,
                max_target_order=MAX_TARGET_ORDER,
                mode=2,
                batch_size=batch_size,
                runtime=runtime,
                descriptor_runner=prepared_runner,
            )

        def prefetched_end_to_end():
            return _run_prepared_batch(
                paths,
                grid_width=1.0,
                max_order=MAX_ORDER,
                max_target_order=MAX_TARGET_ORDER,
                mode=2,
                batch_size=batch_size,
                runtime=runtime,
                descriptor_runner=prepared_runner,
                prefetch=True,
            )

        prepared_runner = _prepare_descriptor_runner(
            max_order=MAX_ORDER,
            max_target_order=MAX_TARGET_ORDER,
            mode=2,
            runtime=runtime,
        )
        sequential_end_to_end()
        batched_end_to_end()
        prefetched_end_to_end()
        end_to_end_samples = _sample_pair(
            sequential_end_to_end,
            batched_end_to_end,
            repeats=repeats,
            sample_count=sample_count,
        )
        prefetch_samples = _sample_pair(
            batched_end_to_end,
            prefetched_end_to_end,
            repeats=repeats,
            sample_count=sample_count,
        )

        padded_cells = int(np.prod(padded_host.shape))
        native_cells = sum(int(np.prod(voxel.shape)) for voxel in host_voxels)
        voxel_dtype = np.dtype(z.FLOAT_DTYPE)
        native_bytes = native_cells * voxel_dtype.itemsize
        padded_bytes = int(padded_host.nbytes)
        host_warmed = {
            name: _summary(samples, batch_size)
            for name, samples in host_prepare_samples.items()
        }
        host_plus_transfer_seconds = sum(
            float(host_warmed[name]["median_seconds"])
            for name in ("parse", "accumulate", "pad")
        ) + float(np.median(transfer_samples[1]))
        accumulate_plus_transfer_seconds = float(
            host_warmed["accumulate"]["median_seconds"]
        ) + float(np.median(transfer_samples[1]))
        checkpoint_0_applicable = BENCHMARK_BACKEND == "gpu" and batch_size == 16
        checkpoint_0_passed = (
            accumulate_plus_transfer_seconds / batch_size >= 0.0005
            and accumulate_plus_transfer_seconds / host_plus_transfer_seconds >= 0.10
        )
        workloads[str(batch_size)] = {
            "input_sequence": [Path(path).stem for path in paths],
            "native_voxel_shapes": [list(voxel.shape) for voxel in host_voxels],
            "padded_voxel_shape": list(padded_host.shape),
            "padding_utilization": native_cells / padded_cells,
            "host_preparation_once_seconds": host_prepare_profile["total_seconds"],
            "host_preparation": {
                "once_seconds": host_prepare_profile["total_seconds"],
                "once_breakdown_seconds": {
                    "parse": host_prepare_profile["parse_seconds"],
                    "accumulate": host_prepare_profile["accumulate_seconds"],
                    "pad": host_prepare_profile["pad_seconds"],
                },
                "warmed": host_warmed,
                "structures": host_prepare_profile["structures"],
            },
            "voxel_storage": {
                "dtype": str(voxel_dtype),
                "native_cells": native_cells,
                "padded_cells": padded_cells,
                "native_bytes": native_bytes,
                "padded_bytes": padded_bytes,
                "padding_bytes": padded_bytes - native_bytes,
                "padding_fraction": 1.0 - native_cells / padded_cells,
            },
            "first_device_execution": {
                "sequential_seconds": sequential_first,
                "batched_seconds": batched_first,
                "whole_pipeline_jit_seconds": compiled_first,
            },
            "transfer": _comparison(*transfer_samples, protein_count=batch_size),
            "transfer_batched": {
                "dtype": str(voxel_dtype),
                "shape": list(padded_host.shape),
                "bytes": padded_bytes,
                "bytes_per_protein": padded_bytes / batch_size,
                "first_seconds": batched_transfer_first,
                "warmed": _summary(transfer_samples[1], batch_size),
            },
            "checkpoint_0_gate": {
                "evaluated": checkpoint_0_applicable,
                "passed": checkpoint_0_passed if checkpoint_0_applicable else None,
                "minimum_accumulate_plus_transfer_milliseconds_per_protein": 0.5,
                "minimum_accumulate_plus_transfer_share": 0.10,
                "accumulate_plus_transfer_milliseconds_per_protein": (
                    1000.0 * accumulate_plus_transfer_seconds / batch_size
                ),
                "host_preparation_plus_transfer_milliseconds_per_protein": (
                    1000.0 * host_plus_transfer_seconds / batch_size
                ),
                "accumulate_plus_transfer_share": (
                    accumulate_plus_transfer_seconds / host_plus_transfer_seconds
                ),
            },
            "device_core": _comparison(*core_samples, protein_count=batch_size),
            "compiled_device_core": _comparison(
                *compiled_core_samples, protein_count=batch_size
            ),
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
            "normalization_method_profile": {
                name: _summary(samples, batch_size)
                for name, samples in normalization_method_samples.items()
            },
            "candidate_solver_profile": {
                "candidate_generation": {
                    name: _summary(samples, batch_size)
                    for name, samples in candidate_solver_samples.items()
                },
                "odd_fused_normalization": {
                    name: _summary(samples, batch_size)
                    for name, samples in odd_normalization_samples.items()
                },
            },
            "rotation_layout_profile": {
                name: _summary(samples, batch_size)
                for name, samples in rotation_layout_samples.items()
            },
            "prepared_end_to_end": _comparison(
                *end_to_end_samples, protein_count=batch_size
            ),
            "host_prefetch": _comparison(
                *prefetch_samples, protein_count=batch_size
            ),
        }

    payload = {
        "schema_version": 8,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_mode": "informational",
        "configuration": {
            "batch_sizes": list(batch_sizes),
            "inputs": [Path(path).stem for path in fixture_paths],
            "input_manifest": os.getenv("ZMPY3D_BATCH_INPUT_MANIFEST"),
            "grid_width": 1.0,
            "max_order": MAX_ORDER,
            "max_target_order": MAX_TARGET_ORDER,
            "mode": 2,
            "samples": sample_count,
            "repeats_per_sample": repeats,
            "jax_x64_enabled": bool(jax.config.x64_enabled),
            "configured_float_dtype": str(z.FLOAT_DTYPE),
            "moment_precision": "auto_mixed_for_float32_order20_plus",
            "jax_requested_backend": BENCHMARK_BACKEND,
            "voxelization": "sequential_numpy_host",
            "padding": "input_order_chunk_high_side_zero_padding",
            "stage_profile": "device_resident_block_after_each_stage",
            "normalization_stage_representation": "analytic_compact",
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
