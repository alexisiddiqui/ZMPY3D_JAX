"""Correct CPU performance comparison against the original NumPy pipeline."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jaxlib
import numpy as np
import pytest

import ZMPY3D_JAX as z

z.configure_for_scientific_computing(enable_x64=True, platform="cpu")

from ZMPY3D_JAX.tests.utils.upstream_regression import (
    REPO_ROOT,
    block_tree,
    load_cache,
    pdb_input,
    run_pipeline,
)


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _time_batch(function, repeats: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(repeats):
        result = function()
        block_tree(result)
    return (time.perf_counter_ns() - start) / repeats / 1e9


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    samples = np.asarray(values, dtype=np.float64)
    return {
        "samples_seconds": samples.tolist(),
        "median_seconds": float(np.median(samples)),
        "p25_seconds": float(np.percentile(samples, 25)),
        "p75_seconds": float(np.percentile(samples, 75)),
        "min_seconds": float(np.min(samples)),
        "max_seconds": float(np.max(samples)),
    }


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


@pytest.mark.filterwarnings("ignore:numpy.fix is deprecated:DeprecationWarning")
def test_regression_runtime_snapshot() -> None:
    """Record cold and warmed timings from two independent CPU pipelines."""
    assert jax.default_backend() == "cpu"
    repeats = _positive_env_int("ZMPY3D_REGRESSION_REPEATS", 3)
    sample_count = _positive_env_int("ZMPY3D_REGRESSION_SAMPLES", 7)
    case = pdb_input(REPO_ROOT / "6NT5.pdb")

    # Materialize setup outside every timed region.
    load_cache(case.max_order)

    def run_jax():
        return run_pipeline(
            "jax", case, normalization_orders=(5,), all_candidates=False
        )

    def run_upstream():
        return run_pipeline(
            "upstream", case, normalization_orders=(5,), all_candidates=False
        )

    jax.clear_caches()
    cold_start = time.perf_counter_ns()
    cold_result = run_jax()
    block_tree(cold_result)
    cold_jax_seconds = (time.perf_counter_ns() - cold_start) / 1e9

    # One untimed warm-up for each implementation before sampling.
    jax_result = run_jax()
    upstream_result = run_upstream()
    assert np.asarray(jax_result["descriptor"]).shape == np.asarray(
        upstream_result["descriptor"]
    ).shape
    assert len(jax_result["rotated"]) == len(upstream_result["rotated"])

    jax_samples: list[float] = []
    upstream_samples: list[float] = []
    for sample_index in range(sample_count):
        # Alternate order to reduce systematic drift from CPU temperature/load.
        ordered = (
            ((run_jax, jax_samples), (run_upstream, upstream_samples))
            if sample_index % 2 == 0
            else ((run_upstream, upstream_samples), (run_jax, jax_samples))
        )
        for function, destination in ordered:
            destination.append(_time_batch(function, repeats))

    jax_summary = _summary(jax_samples)
    upstream_summary = _summary(upstream_samples)
    speed_ratio = (
        upstream_summary["median_seconds"] / jax_summary["median_seconds"]
    )
    payload = {
        "schema_version": 1,
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
            "jax_cold_seconds": cold_jax_seconds,
            "jax_warm": jax_summary,
            "upstream": upstream_summary,
            "speedup_upstream_over_jax": speed_ratio,
        },
    }

    output_path = _output_path()
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logging.info("Upstream regression benchmark saved to: %s", output_path)
