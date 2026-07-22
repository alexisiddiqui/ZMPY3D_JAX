"""Optional JAX-versus-pinned-TensorFlow performance benchmark.

The test process only orchestrates isolated workers. TensorFlow is never imported
by the normal test suite or by the JAX worker.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
WORKER = Path(__file__).resolve()
TF_ENV = WORKER.parent / "tensorflow_env"
TF_PYTHON = os.environ.get(
    "ZMPY3D_TF_PYTHON", str(TF_ENV / ".venv" / "bin" / "python")
)
INPUTS = (ROOT / "6NT5.pdb", ROOT / "6NT6.pdb")


def _env(backend: str) -> dict[str, str]:
    values = os.environ.copy()
    values["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if backend == "cpu":
        values["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        values.pop("CUDA_VISIBLE_DEVICES", None)
    return values


def _positive(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_list(plural_name: str, singular_name: str, default: int) -> tuple[int, ...]:
    raw = os.environ.get(plural_name)
    if raw is None:
        return (_positive(singular_name, default),)
    values = tuple(int(value.strip()) for value in raw.split(","))
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{plural_name} must be a comma-separated list of positive integers")
    return values


def _worker_command(
    framework: str,
    backend: str,
    precision: str,
    order: int,
    mode: int,
    batch: int,
    samples: int,
) -> list[str]:
    python = sys.executable if framework == "jax" else TF_PYTHON
    return [
        python,
        str(WORKER),
        "--worker",
        framework,
        backend,
        precision,
        str(order),
        str(mode),
        str(batch),
        str(samples),
        *(str(path) for path in INPUTS),
    ]


def _run_worker(
    framework: str,
    backend: str,
    precision: str,
    order: int,
    mode: int,
    batch: int,
    samples: int,
) -> dict[str, Any]:
    if framework == "tensorflow" and not Path(TF_PYTHON).exists():
        pytest.skip(
            f"TensorFlow benchmark environment missing: {TF_PYTHON}. "
            f"Create it with: uv sync --project {TF_ENV}"
        )
    result = subprocess.run(
        _worker_command(framework, backend, precision, order, mode, batch, samples),
        cwd=ROOT,
        env=_env(backend),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise AssertionError(
            f"{framework} worker failed with exit code {result.returncode}:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise AssertionError(f"invalid {framework} worker output:\n{result.stdout}\n{result.stderr}") from error


def _parity_metrics(
    jax_result: dict[str, Any], tf_result: dict[str, Any], *, matched: bool
) -> dict[str, Any]:
    actual = np.asarray(jax_result["descriptor"], dtype=np.float64)
    expected = np.asarray(tf_result["descriptor"], dtype=np.float64)
    np.testing.assert_equal(actual.shape, expected.shape)
    assert np.all(np.isfinite(actual))
    assert np.all(np.isfinite(expected))
    difference = actual - expected
    denominator = np.maximum(np.abs(expected), 1e-12)
    actual_rows = actual.reshape(actual.shape[0], -1)
    expected_rows = expected.reshape(expected.shape[0], -1)
    norms = np.linalg.norm(actual_rows, axis=1) * np.linalg.norm(expected_rows, axis=1)
    cosines = np.divide(
        np.sum(actual_rows * expected_rows, axis=1),
        norms,
        out=np.ones_like(norms),
        where=norms != 0,
    )
    allclose = bool(np.allclose(actual, expected, rtol=5e-5, atol=5e-6, equal_nan=True))
    if matched:
        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-6, equal_nan=True)
    return {
        "matched_x64": matched,
        "allclose_rtol_5e-5_atol_5e-6": allclose,
        "descriptor_shape": list(actual.shape),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "max_relative_error": float(np.max(np.abs(difference) / denominator)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "minimum_cosine_similarity": float(np.min(cosines)),
    }


def _summary(samples: list[float], proteins: int = 2) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "median_seconds": float(np.median(values)),
        "p25_seconds": float(np.percentile(values, 25)),
        "p75_seconds": float(np.percentile(values, 75)),
        "median_milliseconds_per_protein": float(np.median(values) * 1000 / proteins),
        "median_proteins_per_second": float(proteins / np.median(values)),
    }


def _compare_case(
    backend: str,
    precision: str,
    order: int,
    mode: int,
    batch: int,
    trial: int,
    samples: int,
    tf_result: dict[str, Any],
) -> dict[str, Any]:
    jax_result = _run_worker("jax", backend, precision, order, mode, batch, samples)
    return {
        "configuration": {
            "backend": backend,
            "precision": precision,
            "order": order,
            "mode": mode,
            "workload": batch,
            "trial": trial,
        },
        "jax": {key: value for key, value in jax_result.items() if key != "descriptor"},
        "tensorflow": {key: value for key, value in tf_result.items() if key != "descriptor"},
        "parity": _parity_metrics(jax_result, tf_result, matched=precision == "x64"),
    }


@pytest.mark.benchmark
def test_tensorflow_comparison_schema() -> None:
    backend = os.environ.get("ZMPY3D_TF_BENCHMARK_BACKEND", "cpu")
    if backend not in {"cpu", "gpu"}:
        raise ValueError("ZMPY3D_TF_BENCHMARK_BACKEND must be cpu or gpu")
    orders = _positive_list("ZMPY3D_TF_BENCHMARK_ORDERS", "ZMPY3D_TF_BENCHMARK_ORDER", 6)
    batches = _positive_list("ZMPY3D_TF_BENCHMARK_BATCHES", "ZMPY3D_TF_BENCHMARK_BATCH", 2)
    samples = _positive("ZMPY3D_TF_BENCHMARK_SAMPLES", 3)
    repeats = _positive("ZMPY3D_TF_BENCHMARK_REPEATS", 1)
    precision_views = ("production", "x64")
    cases = []
    for order in orders:
        for batch in batches:
            for trial in range(1, repeats + 1):
                tf_result = _run_worker("tensorflow", backend, "x64", order, 2, batch, samples)
                for precision in precision_views:
                    cases.append(
                        _compare_case(
                            backend, precision, order, 2, batch, trial, samples, tf_result
                        )
                    )
    output_dir = Path(os.environ.get("ZMPY3D_TF_BENCHMARK_OUTPUT", str(ROOT / "ZMPY3D_JAX/tests/benchmark/_tensorflow")))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"tensorflow_comparison_{backend}_{int(time.time())}.json"
    payload = {
        "schema_version": 2,
        "framework": {"jax": "current workspace", "tensorflow": "externals/ZMPY3D_TF@bb57903716c9eb7666ca0bfcfb3af6135a2d63cd"},
        "environment": {
            "python": platform.python_version(),
            "backend": backend,
            "orders": list(orders),
            "workloads": list(batches),
            "samples_per_trial": samples,
            "trials": repeats,
        },
        "results": cases,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _worker_main(argv: list[str]) -> None:
    framework, backend, precision, order, mode, batch, sample_count = argv[2:9]
    paths = [Path(path) for path in argv[9:]]
    if framework == "jax":
        import jax
        import jax.numpy as jnp
        import ZMPY3D_JAX as z
        from ZMPY3D_JAX.ZMPY3D_CLI_BatchZM import _prepare_batch_runtime
        from ZMPY3D_JAX.lib.batched_descriptor import calculate_descriptor_batch_from_voxels, pad_voxel_batch
        from ZMPY3D_JAX.lib.fill_voxel_by_weight_density04 import fill_voxel_by_weight_density_host

        z.configure_for_scientific_computing(enable_x64=precision == "x64", platform=backend)
        runtime = _prepare_batch_runtime(1.0, int(order))

        def prepare() -> Any:
            voxels = []
            for path in paths * ((int(batch) + len(paths) - 1) // len(paths)):
                xyz, residues = z.get_pdb_xyz_ca(str(path))
                voxel, _ = fill_voxel_by_weight_density_host(
                    xyz, residues, runtime.param["residue_weight_map"], 1.0, runtime.residue_box[1.0]
                )
                voxels.append(voxel)
            return jnp.asarray(pad_voxel_batch(voxels[: int(batch)]), dtype=z.FLOAT_DTYPE)

        workload_paths = (paths * ((int(batch) + len(paths) - 1) // len(paths)))[: int(batch)]
        voxels = prepare()
        kwargs = dict(
            max_order=int(order), max_target_order=5, mode=int(mode),
            default_radius_multiplier=runtime.param["default_radius_multiplier"],
            bbox_to_zm_cache=runtime.bbox_to_zm_cache, x64_bbox_to_zm_cache=runtime.x64_bbox_to_zm_cache,
            rotation_cache=runtime.rotation_cache, descriptor_cache=runtime.descriptor_cache,
        )
        runner = jax.jit(partial(calculate_descriptor_batch_from_voxels, **kwargs))
        compile_start = time.perf_counter()
        first = runner(voxels)
        jax.block_until_ready(first)
        compile_seconds = time.perf_counter() - compile_start
        for _ in range(2):
            jax.block_until_ready(runner(voxels))
        samples = []
        for _ in range(int(sample_count)):
            start = time.perf_counter()
            result = runner(voxels)
            jax.block_until_ready(result)
            samples.append(time.perf_counter() - start)
        for _ in range(2):
            jax.block_until_ready(
                z.ZMPY3D_CLI_BatchZM(
                    [str(path) for path in workload_paths], 1.0, int(order), 5, int(mode), int(batch)
                ).values
            )
        e2e_samples = []
        e2e_result = None
        for _ in range(int(sample_count)):
            start = time.perf_counter()
            e2e_result = z.ZMPY3D_CLI_BatchZM(
                [str(path) for path in workload_paths], 1.0, int(order), 5, int(mode), int(batch)
            )
            jax.block_until_ready(e2e_result.values)
            e2e_samples.append(time.perf_counter() - start)
        print(
            json.dumps(
                {
                    "descriptor": np.asarray(result.values).tolist(),
                    "compile_seconds": compile_seconds,
                    "core": _summary(samples, int(batch)),
                    "end_to_end": _summary(e2e_samples, int(batch)),
                    "device": [str(d) for d in jax.devices()],
                    "version": jax.__version__,
                }
            )
        )
        return

    import tensorflow as tf
    if backend == "cpu":
        tf.config.set_visible_devices([], "GPU")
    sys.path.insert(0, str(ROOT / "externals" / "ZMPY3D_TF"))
    import ZMPY3D_TF as z
    from ZMPY3D_TF.ZMPY3D_TF_CLI_BatchZM import core

    cache_dir = Path(z.__file__).parent / "cache_data"
    with (cache_dir / "BinomialCache.pkl").open("rb") as handle:
        binomial = pickle.load(handle)["BinomialCache"]
    with (cache_dir / f"LogG_CLMCache_MaxOrder{int(order):02d}.pkl").open("rb") as handle:
        cache = pickle.load(handle)
    rotation = cache["RotationIndex"]
    tensors = [
        tf.convert_to_tensor(binomial, tf.float64),
        tf.convert_to_tensor(cache["CLMCache"], tf.float64),
        tf.convert_to_tensor(cache["CLMCache3D"], tf.complex128),
        tf.convert_to_tensor(cache["GCache_complex"]),
        tf.convert_to_tensor(cache["GCache_complex_index"]),
        tf.convert_to_tensor(cache["GCache_pqr_linear"]),
        *(tf.convert_to_tensor(np.squeeze(rotation[key][0, 0]) - (1 if key in {"s_id", "IsNLM_Value"} else 0), tf.int64) for key in ("s_id", "n", "l", "m", "mu", "k", "IsNLM_Value")),
    ]
    params = z.get_global_parameter()
    residue_box = z.get_residue_gaussian_density_cache(params)[1.0]
    workload_paths = (paths * ((int(batch) + len(paths) - 1) // len(paths)))[: int(batch)]
    voxels = []
    for path in workload_paths:
        xyz, residues = z.get_pdb_xyz_ca(str(path))
        voxel, _ = z.fill_voxel_by_weight_density(xyz, residues, params["residue_weight_map"], 1.0, residue_box)
        voxels.append(tf.convert_to_tensor(voxel[:], tf.float64))
    voxels = voxels[: int(batch)]
    args = (int(mode), int(order), 5, *tensors)
    compile_start = time.perf_counter()
    first = [core(voxel, *args).numpy() for voxel in voxels]
    compile_seconds = time.perf_counter() - compile_start
    for _ in range(2):
        [core(voxel, *args).numpy() for voxel in voxels]
    samples = []
    for _ in range(int(sample_count)):
        start = time.perf_counter()
        result = [core(voxel, *args).numpy() for voxel in voxels]
        samples.append(time.perf_counter() - start)
    for _ in range(2):
        [value.numpy() for value in z.ZMPY3D_TF_CLI_BatchZM([str(path) for path in workload_paths], 1.0, int(order), 5, int(mode))]
    e2e_samples = []
    e2e_result = None
    for _ in range(int(sample_count)):
        start = time.perf_counter()
        e2e_result = z.ZMPY3D_TF_CLI_BatchZM([str(path) for path in workload_paths], 1.0, int(order), 5, int(mode))
        [value.numpy() for value in e2e_result]
        e2e_samples.append(time.perf_counter() - start)
    print(
        json.dumps(
            {
                "descriptor": np.asarray(result).tolist(),
                "compile_seconds": compile_seconds,
                "core": _summary(samples, int(batch)),
                "end_to_end": _summary(e2e_samples, int(batch)),
                "device": [d.name for d in tf.config.list_logical_devices()],
                "version": tf.__version__,
            }
        )
    )


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--worker":
    import pickle
    _worker_main(sys.argv)
