"""Latency and stage profiling for the superposition workflow."""

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
import jaxlib
import numpy as np

import ZMPY3D_JAX as z

BENCHMARK_BACKEND = os.getenv("ZMPY3D_BENCHMARK_BACKEND", "cpu").lower()
if BENCHMARK_BACKEND not in {"cpu", "gpu"}:
    raise ValueError("ZMPY3D_BENCHMARK_BACKEND must be 'cpu' or 'gpu'")
z.configure_for_scientific_computing(enable_x64=True, platform=BENCHMARK_BACKEND)

from ZMPY3D_JAX.tests.utils.upstream_regression import (  # noqa: E402
    REPO_ROOT,
    block_tree,
)
from ZMPY3D_JAX.lib.superposition import (  # noqa: E402
    assemble_superposition_features,
    calculate_structure_moments,
    calculate_superposition_candidates,
    calculate_superposition_features,
    calculate_superposition_rotations,
    match_superposition_features,
    prepare_superposition_runtime,
    select_superposition_features,
    solve_superposition_transform,
)


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _time_repeated(function: Callable[[], Any], repeats: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(repeats):
        block_tree(function())
    return (time.perf_counter_ns() - start) / repeats / 1e9


def _time_call(function: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter_ns()
    result = function()
    block_tree(result)
    return result, (time.perf_counter_ns() - start) / 1e9


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


def _summary(values: list[float], pair_count: int = 1) -> dict[str, Any]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.size == 0 or np.any(samples <= 0) or not np.all(np.isfinite(samples)):
        raise ValueError("timing samples must be positive, finite, and non-empty")
    median = float(np.median(samples))
    return {
        "samples_seconds": samples.tolist(),
        "median_seconds": median,
        "p25_seconds": float(np.percentile(samples, 25)),
        "p75_seconds": float(np.percentile(samples, 75)),
        "median_milliseconds_per_pair": 1000.0 * median / pair_count,
        "median_pairs_per_second": pair_count / median,
    }


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
    configured = os.getenv("ZMPY3D_SUPERPOSITION_BENCHMARK_OUTPUT") or os.getenv(
        "ZMPY3D_BENCHMARK_OUTPUT"
    )
    output_dir = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parent / "_simple_time_benchmark"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    label = os.getenv("ZMPY3D_SUPERPOSITION_IMPLEMENTATION", "device")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return output_dir / f"superposition_{label}_{BENCHMARK_BACKEND}_{timestamp}.json"


def test_superposition_latency_snapshot() -> None:
    repeats = _positive_env_int("ZMPY3D_SUPERPOSITION_REPEATS", 1)
    sample_count = _positive_env_int("ZMPY3D_SUPERPOSITION_SAMPLES", 3)
    batch_size = _positive_env_int("ZMPY3D_SUPERPOSITION_BATCH_SIZE", 16)
    pdb_a = str(REPO_ROOT / "6NT5.pdb")
    pdb_b = str(REPO_ROOT / "6NT6.pdb")
    batch_a = [pdb_a if index % 2 == 0 else pdb_b for index in range(batch_size)]
    batch_b = [pdb_b if index % 2 == 0 else pdb_a for index in range(batch_size)]

    functions = {
        "identity": lambda: z.ZMPY3D_CLI_SuperA2B(pdb_a, pdb_a),
        "cross": lambda: z.ZMPY3D_CLI_SuperA2B(pdb_a, pdb_b),
        "batch": lambda: z.ZMPY3D_CLI_BatchSuperA2B(batch_a, batch_b),
    }

    first_execution = {}
    for name, function in functions.items():
        jax.clear_caches()
        first_execution[name] = _time_repeated(function, 1)
    for function in functions.values():
        block_tree(function())
    samples = _sample_functions(functions, repeats=repeats, sample_count=sample_count)

    runtime, runtime_setup_seconds = _time_call(prepare_superposition_runtime)
    (center_a, raw_a), _ = _time_call(
        lambda: calculate_structure_moments(pdb_a, runtime)
    )
    (center_b, raw_b), _ = _time_call(
        lambda: calculate_structure_moments(pdb_b, runtime)
    )
    candidates_a = calculate_superposition_candidates(raw_a)
    candidates_b = calculate_superposition_candidates(raw_b)
    rotations_a = calculate_superposition_rotations(raw_a, candidates_a, runtime)
    rotations_b = calculate_superposition_rotations(raw_b, candidates_b, runtime)
    features_a = assemble_superposition_features(
        center_a, candidates_a, rotations_a, runtime.rotation_feature_indices
    )
    features_b = assemble_superposition_features(
        center_b, candidates_b, rotations_b, runtime.rotation_feature_indices
    )
    selection = select_superposition_features(features_a, features_b)
    block_tree((features_a, features_b, selection))

    def structure_to_moments():
        return (
            calculate_structure_moments(pdb_a, runtime),
            calculate_structure_moments(pdb_b, runtime),
        )

    def candidate_generation():
        return (
            calculate_superposition_candidates(raw_a),
            calculate_superposition_candidates(raw_b),
        )

    def rotation():
        return (
            calculate_superposition_rotations(raw_a, candidates_a, runtime),
            calculate_superposition_rotations(raw_b, candidates_b, runtime),
        )

    def feature_assembly():
        return (
            assemble_superposition_features(
                center_a,
                candidates_a,
                rotations_a,
                runtime.rotation_feature_indices,
            ),
            assemble_superposition_features(
                center_b,
                candidates_b,
                rotations_b,
                runtime.rotation_feature_indices,
            ),
        )

    def device_post_moment():
        item_a = calculate_superposition_features(center_a, raw_a, runtime)
        item_b = calculate_superposition_features(center_b, raw_b, runtime)
        return match_superposition_features(item_a, item_b)

    def legacy_host_match():
        valid_a = np.asarray(features_a.is_valid)
        valid_b = np.asarray(features_b.is_valid)
        values_a = np.asarray(features_a.values)[:, valid_a]
        values_b = np.asarray(features_b.values)[:, valid_b]
        pairs_a = np.asarray(features_a.pairs)[valid_a]
        pairs_b = np.asarray(features_b.pairs)[valid_b]
        similarity = np.abs(values_a.conj().T @ values_b)
        index_a, index_b = np.argwhere(similarity == np.max(similarity))[0]
        rotation_a = np.asarray(
            z.get_transform_matrix_from_ab_list(
                pairs_a[index_a, 0], pairs_a[index_a, 1], center_a
            )
        )
        rotation_b = np.asarray(
            z.get_transform_matrix_from_ab_list(
                pairs_b[index_b, 0], pairs_b[index_b, 1], center_b
            )
        )
        return np.linalg.solve(rotation_b, rotation_a)

    def hybrid_host_solve():
        selected = select_superposition_features(features_a, features_b)
        pair_a = features_a.pairs[selected.index_a]
        pair_b = features_b.pairs[selected.index_b]
        rotation_a = np.asarray(
            z.get_transform_matrix_from_ab_list(
                pair_a[0], pair_a[1], features_a.center_scaled
            )
        )
        rotation_b = np.asarray(
            z.get_transform_matrix_from_ab_list(
                pair_b[0], pair_b[1], features_b.center_scaled
            )
        )
        return np.linalg.solve(rotation_b, rotation_a)

    stage_functions = {
        "structure_to_moments": structure_to_moments,
        "candidate_generation": candidate_generation,
        "rotation": rotation,
        "feature_assembly": feature_assembly,
        "similarity_selection": lambda: select_superposition_features(
            features_a, features_b
        ),
        "transform_solve": lambda: solve_superposition_transform(
            features_a, features_b, selection
        ),
        "device_post_moment": device_post_moment,
    }
    for function in stage_functions.values():
        block_tree(function())
    stage_samples = _sample_functions(
        stage_functions, repeats=repeats, sample_count=sample_count
    )
    comparison_functions = {
        "legacy_host": legacy_host_match,
        "hybrid_host_solve": hybrid_host_solve,
        "device": lambda: match_superposition_features(features_a, features_b),
    }
    comparison_results = {
        name: function() for name, function in comparison_functions.items()
    }
    block_tree(comparison_results)
    np.testing.assert_allclose(
        comparison_results["device"].matrix,
        comparison_results["legacy_host"],
        rtol=1e-10,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        comparison_results["device"].matrix,
        comparison_results["hybrid_host_solve"],
        rtol=1e-10,
        atol=1e-8,
    )
    comparison_samples = _sample_functions(
        comparison_functions, repeats=repeats, sample_count=sample_count
    )

    trace_root = os.getenv("ZMPY3D_SUPERPOSITION_TRACE_DIR")
    if trace_root:
        trace_dir = Path(trace_root).expanduser() / BENCHMARK_BACKEND
        trace_dir.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
            for name, function in stage_functions.items():
                with jax.profiler.TraceAnnotation(name):
                    block_tree(function())

    payload = {
        "schema_version": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_mode": "informational",
        "implementation": os.getenv(
            "ZMPY3D_SUPERPOSITION_IMPLEMENTATION", "device"
        ),
        "configuration": {
            "inputs": ["6NT5", "6NT6"],
            "max_order": 6,
            "grid_width": 1.0,
            "batch_size": batch_size,
            "samples": sample_count,
            "repeats_per_sample": repeats,
            "jax_requested_backend": BENCHMARK_BACKEND,
            "jax_x64_enabled": bool(jax.config.x64_enabled),
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
        "results": {
            "runtime_setup_seconds": runtime_setup_seconds,
            "first_execution_seconds": first_execution,
            "warm": {
                "identity": _summary(samples["identity"]),
                "cross": _summary(samples["cross"]),
                "batch": _summary(samples["batch"], pair_count=batch_size),
            },
            "stage_profile": {
                name: _summary(values) for name, values in stage_samples.items()
            },
            "post_moment_comparison": {
                name: _summary(values)
                for name, values in comparison_samples.items()
            },
        },
    }

    output_path = _output_path()
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
