"""CPU latency and stage profiling against the original NumPy pipeline."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import jax
import jaxlib
import numpy as np
import pytest

import ZMPY3D_JAX as z

z.configure_for_scientific_computing(enable_x64=True, platform="cpu")

from ZMPY3D_JAX.tests.utils.upstream_regression import (
    REPO_ROOT,
    PipelineContext,
    block_tree,
    build_jax_setup,
    build_upstream_setup,
    load_cache,
    pdb_input,
    prepare_pipeline_context,
    run_prepared_pipeline,
)


STAGE_NAMES = (
    "voxelization",
    "bbox_order1",
    "radius_and_sphere",
    "bbox_max_order",
    "bbox_to_zm",
    "descriptor",
    "ab_candidates",
    "zm_rotation",
)


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _time_call(function: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter_ns()
    result = function()
    block_tree(result)
    return result, (time.perf_counter_ns() - start) / 1e9


def _time_batch(function: Callable[[], Any], repeats: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(repeats):
        result = function()
        block_tree(result)
    return (time.perf_counter_ns() - start) / repeats / 1e9


def _profile_batch(context: PipelineContext, repeats: int) -> dict[str, float]:
    totals_ns = {stage: 0 for stage in STAGE_NAMES}

    for _ in range(repeats):
        observed: set[str] = set()

        def execute_stage(name: str, function: Callable[[], Any]) -> Any:
            assert name in totals_ns
            assert name not in observed
            observed.add(name)
            start = time.perf_counter_ns()
            result = function()
            block_tree(result)
            totals_ns[name] += time.perf_counter_ns() - start
            return result

        run_prepared_pipeline(
            context,
            normalization_orders=(5,),
            all_candidates=False,
            stage_executor=execute_stage,
        )
        assert observed == set(STAGE_NAMES)

    return {stage: totals_ns[stage] / repeats / 1e9 for stage in STAGE_NAMES}


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.size == 0 or not np.all(np.isfinite(samples)) or np.any(samples < 0):
        raise ValueError("Timing samples must be finite, non-negative, and non-empty")
    return {
        "samples_seconds": samples.tolist(),
        "median_seconds": float(np.median(samples)),
        "p25_seconds": float(np.percentile(samples, 25)),
        "p75_seconds": float(np.percentile(samples, 75)),
        "min_seconds": float(np.min(samples)),
        "max_seconds": float(np.max(samples)),
    }


def _time_ratios(jax_seconds: float, upstream_seconds: float) -> dict[str, float]:
    if jax_seconds <= 0 or upstream_seconds <= 0:
        raise ValueError("Time ratios require positive timings")
    return {
        "jax_over_upstream_time_ratio": jax_seconds / upstream_seconds,
        "upstream_over_jax_time_ratio": upstream_seconds / jax_seconds,
    }


def _summarize_stages(
    samples: dict[str, list[float]],
) -> dict[str, dict[str, float | list[float]]]:
    assert set(samples) == set(STAGE_NAMES)
    return {stage: _summary(samples[stage]) for stage in STAGE_NAMES}


def _rank_bottlenecks(
    jax_stages: dict[str, dict[str, Any]],
    upstream_stages: dict[str, dict[str, Any]],
) -> list[dict[str, float | int | str]]:
    rows = []
    for stage in STAGE_NAMES:
        jax_median = float(jax_stages[stage]["median_seconds"])
        upstream_median = float(upstream_stages[stage]["median_seconds"])
        excess = jax_median - upstream_median
        rows.append(
            {
                "stage": stage,
                "jax_median_seconds": jax_median,
                "upstream_median_seconds": upstream_median,
                "median_excess_seconds": excess,
                **_time_ratios(jax_median, upstream_median),
            }
        )

    rows.sort(key=lambda row: float(row["median_excess_seconds"]), reverse=True)
    positive_total = sum(max(float(row["median_excess_seconds"]), 0.0) for row in rows)
    for rank, row in enumerate(rows, start=1):
        positive_excess = max(float(row["median_excess_seconds"]), 0.0)
        row["rank"] = rank
        row["positive_excess_contribution_percent"] = (
            100.0 * positive_excess / positive_total if positive_total > 0 else 0.0
        )
    return rows


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


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
    configured = os.getenv("ZMPY3D_BENCHMARK_OUTPUT")
    output_dir = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parent / "_simple_time_benchmark"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return output_dir / f"upstream_regression_benchmark_{timestamp}.json"


def _validate_payload(payload: dict[str, Any]) -> None:
    assert payload["schema_version"] == 2
    results = payload["results"]
    assert set(results) == {
        "setup",
        "first_execution",
        "warm_end_to_end",
        "stage_profile",
    }
    for section in ("setup", "first_execution"):
        numeric = [value for value in results[section].values() if isinstance(value, (int, float))]
        assert numeric and all(np.isfinite(value) and value >= 0 for value in numeric)

    warm = results["warm_end_to_end"]
    assert warm["jax_over_upstream_time_ratio"] > 0
    assert warm["upstream_over_jax_time_ratio"] > 0
    np.testing.assert_allclose(
        warm["jax_over_upstream_time_ratio"]
        * warm["upstream_over_jax_time_ratio"],
        1.0,
    )

    stage_profile = results["stage_profile"]
    assert set(stage_profile["jax"]) == set(STAGE_NAMES)
    assert set(stage_profile["upstream"]) == set(STAGE_NAMES)
    assert len(stage_profile["ranking"]) == len(STAGE_NAMES)
    assert stage_profile["bottleneck"] == stage_profile["ranking"][0]


def test_summary_statistics_and_ratio_direction() -> None:
    summary = _summary([1.0, 2.0, 3.0])
    assert summary["median_seconds"] == 2.0
    assert summary["p25_seconds"] == 1.5
    assert summary["p75_seconds"] == 2.5
    ratios = _time_ratios(5.0, 2.0)
    assert ratios == {
        "jax_over_upstream_time_ratio": 2.5,
        "upstream_over_jax_time_ratio": 0.4,
    }


def test_bottleneck_ranking_uses_positive_median_excess() -> None:
    jax_stages = {
        stage: {"median_seconds": float(index + 2)}
        for index, stage in enumerate(STAGE_NAMES)
    }
    upstream_stages = {
        stage: {"median_seconds": float(index + 1)}
        for index, stage in enumerate(STAGE_NAMES)
    }
    jax_stages["bbox_to_zm"]["median_seconds"] = 20.0
    ranking = _rank_bottlenecks(jax_stages, upstream_stages)
    assert ranking[0]["stage"] == "bbox_to_zm"
    assert ranking[0]["rank"] == 1
    np.testing.assert_allclose(
        sum(row["positive_excess_contribution_percent"] for row in ranking), 100.0
    )


@pytest.mark.filterwarnings("ignore:numpy.fix is deprecated:DeprecationWarning")
def test_regression_runtime_snapshot() -> None:
    """Record setup, first-execution, warmed latency, and stage-level CPU timings."""
    assert jax.default_backend() == "cpu"
    repeats = _positive_env_int("ZMPY3D_REGRESSION_REPEATS", 3)
    sample_count = _positive_env_int("ZMPY3D_REGRESSION_SAMPLES", 7)

    jax.clear_caches()
    load_cache.cache_clear()
    shared_start = time.perf_counter_ns()
    case = pdb_input(REPO_ROOT / "6NT5.pdb")
    cache = load_cache(case.max_order)
    block_tree((case.xyz, cache))
    shared_setup_seconds = (time.perf_counter_ns() - shared_start) / 1e9

    jax_setup_value, jax_setup_seconds = _time_call(build_jax_setup)
    upstream_setup_value, upstream_setup_seconds = _time_call(build_upstream_setup)
    jax_context = prepare_pipeline_context(
        "jax", case, cache=cache, setup=jax_setup_value
    )
    upstream_context = prepare_pipeline_context(
        "upstream", case, cache=cache, setup=upstream_setup_value
    )

    def run_jax():
        return run_prepared_pipeline(
            jax_context, normalization_orders=(5,), all_candidates=False
        )

    def run_upstream():
        return run_prepared_pipeline(
            upstream_context, normalization_orders=(5,), all_candidates=False
        )

    jax.clear_caches()
    jax_first_result, jax_first_seconds = _time_call(run_jax)
    upstream_first_result, upstream_first_seconds = _time_call(run_upstream)
    assert np.asarray(jax_first_result["descriptor"]).shape == np.asarray(
        upstream_first_result["descriptor"]
    ).shape
    assert len(jax_first_result["rotated"]) == len(upstream_first_result["rotated"])

    # Additional untimed calls establish a common warmed state before sampling.
    run_jax()
    run_upstream()

    jax_samples: list[float] = []
    upstream_samples: list[float] = []
    for sample_index in range(sample_count):
        ordered = (
            ((run_jax, jax_samples), (run_upstream, upstream_samples))
            if sample_index % 2 == 0
            else ((run_upstream, upstream_samples), (run_jax, jax_samples))
        )
        for function, destination in ordered:
            destination.append(_time_batch(function, repeats))

    jax_stage_samples = {stage: [] for stage in STAGE_NAMES}
    upstream_stage_samples = {stage: [] for stage in STAGE_NAMES}
    for sample_index in range(sample_count):
        ordered_contexts = (
            ((jax_context, jax_stage_samples), (upstream_context, upstream_stage_samples))
            if sample_index % 2 == 0
            else ((upstream_context, upstream_stage_samples), (jax_context, jax_stage_samples))
        )
        for context, destination in ordered_contexts:
            batch = _profile_batch(context, repeats)
            for stage in STAGE_NAMES:
                destination[stage].append(batch[stage])

    jax_summary = _summary(jax_samples)
    upstream_summary = _summary(upstream_samples)
    jax_stage_summary = _summarize_stages(jax_stage_samples)
    upstream_stage_summary = _summarize_stages(upstream_stage_samples)
    ranking = _rank_bottlenecks(jax_stage_summary, upstream_stage_summary)
    warm_ratios = _time_ratios(
        float(jax_summary["median_seconds"]),
        float(upstream_summary["median_seconds"]),
    )

    payload = {
        "schema_version": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_mode": "informational",
        "configuration": {
            "input": case.name,
            "residue_count": len(case.residues),
            "grid_width": case.grid_width,
            "max_order": case.max_order,
            "normalization_orders": [5],
            "samples": sample_count,
            "repeats_per_sample": repeats,
            "jax_x64_enabled": bool(jax.config.x64_enabled),
        },
        "environment": {
            "git_revision": _git_revision(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": _cpu_model(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "results": {
            "setup": {
                "shared_input_and_cache_seconds": shared_setup_seconds,
                "jax_seconds": jax_setup_seconds,
                "upstream_seconds": upstream_setup_seconds,
            },
            "first_execution": {
                "jax_seconds": jax_first_seconds,
                "upstream_seconds": upstream_first_seconds,
            },
            "warm_end_to_end": {
                "jax": jax_summary,
                "upstream": upstream_summary,
                **warm_ratios,
            },
            "stage_profile": {
                "synchronization": "block_after_each_stage",
                "note": "Synchronized stage medians are diagnostic and do not sum to end-to-end latency.",
                "jax": jax_stage_summary,
                "upstream": upstream_stage_summary,
                "ranking": ranking,
                "bottleneck": ranking[0],
            },
        },
    }
    _validate_payload(payload)

    output_path = _output_path()
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logging.info("Upstream regression benchmark saved to: %s", output_path)
    logging.info(
        "Top CPU bottleneck: %s (%.3f ms median excess, %.1f%% of positive excess)",
        ranking[0]["stage"],
        1000 * float(ranking[0]["median_excess_seconds"]),
        float(ranking[0]["positive_excess_contribution_percent"]),
    )
