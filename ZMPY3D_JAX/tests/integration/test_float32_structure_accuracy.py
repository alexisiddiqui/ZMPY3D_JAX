"""Cross-precision characterization of descriptor structure discrimination."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


STRUCTURES = ("6NT5", "6NT6")
TARGET_ORDERS = (2, 3, 4, 5)
PRECISION_FRONTIERS = ("cartesian_x64", "moments_x64")
SCHEMA_VERSION = 2


def _run_worker(
    max_order: int,
    backend: str,
    enable_x64: bool,
    output: Path,
    reference_input: Path | None,
) -> None:
    import jax
    import jax.numpy as jnp

    import ZMPY3D_JAX as z

    z.configure_for_scientific_computing(enable_x64=enable_x64, platform=backend)
    if not enable_x64:
        # Retain float32 library defaults while permitting explicit prototype x64 ops.
        jax.config.update("jax_enable_x64", True)

    from ZMPY3D_JAX.ZMPY3D_CLI_BatchZM import _prepare_batch_runtime
    from ZMPY3D_JAX.lib.batched_descriptor import (
        _assemble_descriptor_batch,
        _calculate_3dzd_batch,
        _calculate_ab_compact_candidates_batch,
        _calculate_bbox_max_order_batch,
        _calculate_bbox_order1_batch,
        _calculate_bbox_to_zm_batch,
        _calculate_normalization_means,
        _calculate_radius_and_samples_batch,
        calculate_descriptor_batch_from_voxels,
        pad_voxel_batch,
    )
    from ZMPY3D_JAX.lib.fill_voxel_by_weight_density04 import (
        fill_voxel_by_weight_density_host,
    )
    from ZMPY3D_JAX.lib.calculate_bbox_moment_2_zm05 import BBoxToZMCache
    from ZMPY3D_JAX.lib.mixed_precision_prototype import (
        calculate_descriptor_mixed_prototype,
    )

    runtime = _prepare_batch_runtime(1.0, max_order)
    repo_root = Path(__file__).resolve().parents[3]
    with (
        repo_root
        / "ZMPY3D_JAX"
        / "cache_data"
        / f"LogG_CLMCache_MaxOrder{max_order:02d}.pkl"
    ).open("rb") as handle:
        raw_cache = pickle.load(handle)
    x64_bbox_cache = BBoxToZMCache(
        max_order=max_order,
        g_coefficients=jnp.asarray(
            raw_cache["GCache_complex"], dtype=jnp.complex128
        ).reshape(-1),
        pqr_indices=jnp.asarray(
            raw_cache["GCache_pqr_linear"], dtype=jnp.int32
        ).reshape(-1)
        - 1,
        output_indices=jnp.asarray(
            raw_cache["GCache_complex_index"], dtype=jnp.int32
        ).reshape(-1)
        - 1,
        clm=jnp.asarray(raw_cache["CLMCache3D"], dtype=jnp.complex128),
    )
    host_voxels = []
    for structure in STRUCTURES:
        xyz, residues = z.get_pdb_xyz_ca(str(repo_root / f"{structure}.pdb"))
        voxel, _ = fill_voxel_by_weight_density_host(
            xyz,
            residues,
            runtime.param["residue_weight_map"],
            1.0,
            runtime.residue_box[1.0],
        )
        host_voxels.append(voxel)

    dtype = jnp.float64 if enable_x64 else jnp.float32
    voxels = jnp.asarray(pad_voxel_batch(host_voxels), dtype=dtype)
    masses, centers, order1_moments = _calculate_bbox_order1_batch(voxels)
    radius = _calculate_radius_and_samples_batch(
        voxels,
        centers,
        masses,
        runtime.param["default_radius_multiplier"],
    )
    _, average_radius, max_radius, x_samples, y_samples, z_samples = radius
    _, _, bbox_moments = _calculate_bbox_max_order_batch(
        voxels, max_order, x_samples, y_samples, z_samples
    )
    scaled, raw = _calculate_bbox_to_zm_batch(
        bbox_moments,
        max_order,
        runtime.bbox_to_zm_cache.g_coefficients,
        runtime.bbox_to_zm_cache.pqr_indices,
        runtime.bbox_to_zm_cache.output_indices,
        runtime.bbox_to_zm_cache.clm,
    )
    descriptors_3dzd = _calculate_3dzd_batch(scaled)
    means = _calculate_normalization_means(
        raw,
        TARGET_ORDERS,
        "companion_compact_grouped",
        runtime.rotation_cache,
        "auto",
    )
    descriptor = _assemble_descriptor_batch(
        descriptors_3dzd,
        means,
        runtime.descriptor_cache.descriptor_indices,
        runtime.descriptor_cache.moment_indices,
    )
    candidate_counts = jnp.stack(
        [
            jnp.sum(
                _calculate_ab_compact_candidates_batch(raw, order).is_valid,
                axis=1,
            )
            for order in TARGET_ORDERS
        ],
        axis=1,
    )

    descriptor_3dzd_values = descriptors_3dzd.reshape((2, -1))[
        :, runtime.descriptor_cache.descriptor_indices
    ]
    moment_size = means.shape[2] * means.shape[3] * means.shape[4]
    normalization_values = means.reshape((2, len(TARGET_ORDERS), moment_size))[
        :, :, runtime.descriptor_cache.moment_indices
    ]
    radius_values = jnp.concatenate(
        (
            masses[:, None],
            centers,
            average_radius[:, None],
            max_radius[:, None],
            x_samples,
            y_samples,
            z_samples,
        ),
        axis=1,
    )
    order1_values = jnp.concatenate(
        (masses[:, None], centers, order1_moments.reshape((2, -1))), axis=1
    )

    payload = {
        "voxels": voxels,
        "x_samples": x_samples,
        "y_samples": y_samples,
        "z_samples": z_samples,
        "bbox_order1": order1_values,
        "radius_and_samples": radius_values,
        "bbox_max_order": bbox_moments,
        "zm_raw": raw,
        "zm_scaled": scaled,
        "descriptor_3dzd": descriptor_3dzd_values,
        "normalization_all": normalization_values,
        "descriptor_final": descriptor.values,
        "descriptor_valid": descriptor.is_valid,
        "candidate_counts": candidate_counts,
    }
    for index, order in enumerate(TARGET_ORDERS):
        payload[f"normalization_order_{order}"] = normalization_values[:, index]
    if max_order == 20:
        properties = z.get_descriptor_property()
        payload["score_indices"] = jnp.concatenate(
            [properties[f"ZMIndex{index}"] for index in range(5)]
        ).reshape(-1)
        payload["score_weights"] = jnp.concatenate(
            [properties[f"ZMWeight{index}"] for index in range(5)]
        ).reshape(-1)

    if not enable_x64:
        prototype_descriptors = {}
        for frontier in PRECISION_FRONTIERS:
            (
                prototype_descriptor,
                prototype_bbox,
                prototype_scaled,
                prototype_raw,
                prototype_3dzd,
                prototype_means,
            ) = calculate_descriptor_mixed_prototype(
                voxels,
                max_order=max_order,
                target_orders=TARGET_ORDERS,
                x_samples=x_samples,
                y_samples=y_samples,
                z_samples=z_samples,
                configured_bbox_cache=runtime.bbox_to_zm_cache,
                x64_bbox_cache=x64_bbox_cache,
                rotation_cache=runtime.rotation_cache,
                descriptor_cache=runtime.descriptor_cache,
                precision_frontier=frontier,
            )
            prototype_3dzd_values = prototype_3dzd.reshape((2, -1))[
                :, runtime.descriptor_cache.descriptor_indices
            ]
            prototype_normalization = prototype_means.reshape(
                (2, len(TARGET_ORDERS), moment_size)
            )[:, :, runtime.descriptor_cache.moment_indices]
            prototype_candidate_counts = jnp.stack(
                [
                    jnp.sum(
                        _calculate_ab_compact_candidates_batch(
                            prototype_raw, order
                        ).is_valid,
                        axis=1,
                    )
                    for order in TARGET_ORDERS
                ],
                axis=1,
            )
            prefix = f"prototype_{frontier}"
            prototype_descriptors[frontier] = prototype_descriptor
            payload.update(
                {
                    f"{prefix}_bbox_max_order": prototype_bbox,
                    f"{prefix}_zm_raw": prototype_raw,
                    f"{prefix}_zm_scaled": prototype_scaled,
                    f"{prefix}_descriptor_3dzd": prototype_3dzd_values,
                    f"{prefix}_normalization_all": prototype_normalization,
                    f"{prefix}_descriptor_final": prototype_descriptor.values,
                    f"{prefix}_descriptor_valid": prototype_descriptor.is_valid,
                    f"{prefix}_candidate_counts": prototype_candidate_counts,
                }
            )
            if max_order == 20:
                repeated_descriptor = calculate_descriptor_mixed_prototype(
                    voxels,
                    max_order=max_order,
                    target_orders=TARGET_ORDERS,
                    x_samples=x_samples,
                    y_samples=y_samples,
                    z_samples=z_samples,
                    configured_bbox_cache=runtime.bbox_to_zm_cache,
                    x64_bbox_cache=x64_bbox_cache,
                    rotation_cache=runtime.rotation_cache,
                    descriptor_cache=runtime.descriptor_cache,
                    precision_frontier=frontier,
                )[0]
                np.testing.assert_array_equal(
                    np.asarray(repeated_descriptor.values),
                    np.asarray(prototype_descriptor.values),
                )
        production_descriptor = calculate_descriptor_batch_from_voxels(
            voxels,
            max_order=max_order,
            max_target_order=5,
            mode=2,
            default_radius_multiplier=runtime.param[
                "default_radius_multiplier"
            ],
            bbox_to_zm_cache=runtime.bbox_to_zm_cache,
            x64_bbox_to_zm_cache=x64_bbox_cache,
            rotation_cache=runtime.rotation_cache,
            x64_rotation_cache=runtime.x64_rotation_cache,
            descriptor_cache=runtime.descriptor_cache,
        )
        assert production_descriptor.values.dtype == dtype
        payload["production_descriptor_final"] = production_descriptor.values
        payload["production_descriptor_valid"] = production_descriptor.is_valid

    if reference_input is not None:
        with np.load(reference_input) as reference:
            reference_voxels = jnp.asarray(reference["voxels"], dtype=dtype)
            reference_x = jnp.asarray(reference["x_samples"], dtype=dtype)
            reference_y = jnp.asarray(reference["y_samples"], dtype=dtype)
            reference_z = jnp.asarray(reference["z_samples"], dtype=dtype)
            reference_bbox = jnp.asarray(reference["bbox_max_order"], dtype=dtype)
            reference_raw = jnp.asarray(reference["zm_raw"])
            reference_scaled = jnp.asarray(reference["zm_scaled"])

        isolated_bbox = _calculate_bbox_max_order_batch(
            reference_voxels,
            max_order,
            reference_x,
            reference_y,
            reference_z,
        )[2]
        isolated_scaled, isolated_raw = _calculate_bbox_to_zm_batch(
            reference_bbox,
            max_order,
            runtime.bbox_to_zm_cache.g_coefficients,
            runtime.bbox_to_zm_cache.pqr_indices,
            runtime.bbox_to_zm_cache.output_indices,
            runtime.bbox_to_zm_cache.clm,
        )
        isolated_3dzd = _calculate_3dzd_batch(reference_scaled)
        isolated_means = _calculate_normalization_means(
            reference_raw,
            TARGET_ORDERS,
            "companion_compact_grouped",
            runtime.rotation_cache,
            "auto",
        )
        isolated_descriptor = _assemble_descriptor_batch(
            isolated_3dzd,
            isolated_means,
            runtime.descriptor_cache.descriptor_indices,
            runtime.descriptor_cache.moment_indices,
        )
        payload.update(
            {
                "isolated_bbox_max_order": isolated_bbox,
                "isolated_zm_raw": isolated_raw,
                "isolated_zm_scaled": isolated_scaled,
                "isolated_descriptor_3dzd": isolated_3dzd.reshape((2, -1))[
                    :, runtime.descriptor_cache.descriptor_indices
                ],
                "isolated_normalization_all": isolated_means.reshape(
                    (2, len(TARGET_ORDERS), moment_size)
                )[:, :, runtime.descriptor_cache.moment_indices],
                "isolated_descriptor_final": isolated_descriptor.values,
            }
        )

    jax.block_until_ready(payload)
    np.savez_compressed(output, **{key: np.asarray(value) for key, value in payload.items()})


def _finite_rows(reference: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference).reshape((2, -1))
    target = np.asarray(target).reshape((2, -1))
    finite = np.all(np.isfinite(reference) & np.isfinite(target), axis=0)
    return reference[:, finite], target[:, finite]


def _norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(values.reshape(-1)))


def _stage_metrics(reference: np.ndarray, target: np.ndarray) -> dict[str, object]:
    reference, target = _finite_rows(reference, target)
    if reference.shape[1] == 0:
        raise AssertionError("stage has no common finite values")

    reference_delta = reference[0] - reference[1]
    target_delta = target[0] - target[1]
    reference_separation = _norm(reference_delta)
    target_separation = _norm(target_delta)
    delta_error = _norm(target_delta - reference_delta)
    denominator = reference_separation * target_separation
    cosine = (
        float(np.real(np.vdot(reference_delta, target_delta)) / denominator)
        if denominator > 0
        else float("nan")
    )

    per_structure = []
    for index, structure in enumerate(STRUCTURES):
        difference = np.abs(target[index] - reference[index])
        error_norm = _norm(difference)
        reference_norm = _norm(reference[index])
        per_structure.append(
            {
                "structure": structure,
                "max_absolute_error": float(np.max(difference)),
                "rms_error": float(np.sqrt(np.mean(difference**2))),
                "relative_l2_error": error_norm / max(reference_norm, np.finfo(float).tiny),
                "self_error_over_reference_separation": (
                    error_norm / reference_separation
                    if reference_separation > 0
                    else float("nan")
                ),
            }
        )

    metrics: dict[str, object] = {
        "comparable_value_count": int(reference.shape[1]),
        "per_structure": per_structure,
        "reference_structure_separation_l2": reference_separation,
        "float32_structure_separation_l2": target_separation,
        "float32_over_reference_separation": (
            target_separation / reference_separation
            if reference_separation > 0
            else float("nan")
        ),
        "difference_vector_relative_l2_error": (
            delta_error / reference_separation
            if reference_separation > 0
            else float("nan")
        ),
        "difference_vector_cosine": float(np.clip(cosine, -1.0, 1.0)),
    }
    if not np.iscomplexobj(reference):
        meaningful = np.abs(reference_delta) > np.finfo(reference.dtype).eps
        metrics["difference_sign_agreement"] = (
            float(np.mean(np.sign(reference_delta[meaningful]) == np.sign(target_delta[meaningful])))
            if np.any(meaningful)
            else float("nan")
        )
    return metrics


def _weighted_score_metrics(
    reference, target, target_descriptor_key: str = "descriptor_final"
) -> dict[str, float]:
    indices = np.asarray(reference["score_indices"], dtype=np.int64)
    weights = np.asarray(reference["score_weights"], dtype=np.float64)
    reference_values = np.asarray(reference["descriptor_final"])
    target_values = np.asarray(target[target_descriptor_key])

    def distance(values_a: np.ndarray, values_b: np.ndarray) -> float:
        return float(np.sum(np.abs(values_a[indices] - values_b[indices]) * weights))

    reference_distance = distance(reference_values[0], reference_values[1])
    target_distance = distance(target_values[0], target_values[1])
    self_errors = [
        distance(reference_values[index], target_values[index]) for index in range(2)
    ]
    return {
        "reference_weighted_distance": reference_distance,
        "float32_weighted_distance": target_distance,
        "weighted_distance_absolute_error": abs(target_distance - reference_distance),
        "reference_similarity_score": (9.0 - reference_distance) / 9.0 * 100.0,
        "float32_similarity_score": (9.0 - target_distance) / 9.0 * 100.0,
        "similarity_score_absolute_error": abs(target_distance - reference_distance)
        / 9.0
        * 100.0,
        "6NT5_self_error_over_reference_distance": self_errors[0]
        / reference_distance,
        "6NT6_self_error_over_reference_distance": self_errors[1]
        / reference_distance,
    }


def _compare(max_order: int, backend: str, directory: Path) -> dict[str, object]:
    reference_path = directory / f"order{max_order}_x64_cpu.npz"
    target_path = directory / f"order{max_order}_float32_{backend}.npz"
    subprocess.run(
        [sys.executable, __file__, "--worker", str(max_order), "cpu", "x64", str(reference_path)],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--worker",
            str(max_order),
            backend,
            "float32",
            str(target_path),
            str(reference_path),
        ],
        check=True,
    )

    with np.load(reference_path) as reference, np.load(target_path) as target:
        np.testing.assert_array_equal(reference["descriptor_valid"], target["descriptor_valid"])
        np.testing.assert_array_equal(reference["candidate_counts"], target["candidate_counts"])
        stage_names = (
            "voxels",
            "bbox_order1",
            "radius_and_samples",
            "bbox_max_order",
            "zm_raw",
            "zm_scaled",
            "descriptor_3dzd",
            "normalization_all",
            *(f"normalization_order_{order}" for order in TARGET_ORDERS),
            "descriptor_final",
        )
        stage_metrics = {
            stage: _stage_metrics(reference[stage], target[stage]) for stage in stage_names
        }
        isolated_stage_pairs = {
            "bbox_max_order": "isolated_bbox_max_order",
            "zm_raw": "isolated_zm_raw",
            "zm_scaled": "isolated_zm_scaled",
            "descriptor_3dzd": "isolated_descriptor_3dzd",
            "normalization_all": "isolated_normalization_all",
            "descriptor_final": "isolated_descriptor_final",
        }
        isolated_metrics = {
            stage: _stage_metrics(reference[stage], target[target_stage])
            for stage, target_stage in isolated_stage_pairs.items()
        }
        prototype_frontiers = {}
        for frontier in PRECISION_FRONTIERS:
            prefix = f"prototype_{frontier}"
            np.testing.assert_array_equal(
                reference["descriptor_valid"], target[f"{prefix}_descriptor_valid"]
            )
            np.testing.assert_array_equal(
                reference["candidate_counts"], target[f"{prefix}_candidate_counts"]
            )
            prototype_stage_pairs = {
                "bbox_max_order": f"{prefix}_bbox_max_order",
                "zm_raw": f"{prefix}_zm_raw",
                "zm_scaled": f"{prefix}_zm_scaled",
                "descriptor_3dzd": f"{prefix}_descriptor_3dzd",
                "normalization_all": f"{prefix}_normalization_all",
                "descriptor_final": f"{prefix}_descriptor_final",
            }
            prototype_result: dict[str, object] = {
                "dtypes": {
                    "bbox_max_order": str(target[f"{prefix}_bbox_max_order"].dtype),
                    "zm_raw": str(target[f"{prefix}_zm_raw"].dtype),
                    "descriptor_final": str(
                        target[f"{prefix}_descriptor_final"].dtype
                    ),
                },
                "stages": {
                    stage: _stage_metrics(reference[stage], target[target_stage])
                    for stage, target_stage in prototype_stage_pairs.items()
                },
            }
            if max_order == 20:
                prototype_result["production_weighted_score"] = (
                    _weighted_score_metrics(
                        reference, target, f"{prefix}_descriptor_final"
                    )
                )
            prototype_frontiers[frontier] = prototype_result
        result: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "reference": {"precision": "x64", "backend": "cpu"},
            "target": {"precision": "float32", "backend": backend},
            "configuration": {
                "structures": list(STRUCTURES),
                "grid_width": 1.0,
                "max_order": max_order,
                "normalization_orders": list(TARGET_ORDERS),
            },
            "candidate_valid_counts": np.asarray(target["candidate_counts"]).tolist(),
            "stages": stage_metrics,
            "isolated_stages": isolated_metrics,
            "prototype_frontiers": prototype_frontiers,
        }
        if max_order == 20:
            np.testing.assert_array_equal(
                reference["descriptor_valid"],
                target["production_descriptor_valid"],
            )
            result["production_descriptor"] = _stage_metrics(
                reference["descriptor_final"],
                target["production_descriptor_final"],
            )
            result["production_weighted_score"] = _weighted_score_metrics(
                reference, target, "production_descriptor_final"
            )

    for metrics in result["stages"].values():
        assert metrics["reference_structure_separation_l2"] > 0
        assert metrics["float32_structure_separation_l2"] > 0
        for value in metrics.values():
            if isinstance(value, float):
                assert np.isfinite(value)
    return result


def _apply_cpu_accuracy_gate(
    gpu_result: dict[str, object], cpu_result: dict[str, object]
) -> None:
    cpu_descriptor = cpu_result["stages"]["descriptor_final"]
    cpu_score_error = cpu_result["production_weighted_score"][
        "similarity_score_absolute_error"
    ]
    thresholds = {
        "maximum_similarity_score_absolute_error": cpu_score_error,
        "maximum_difference_vector_relative_l2_error": cpu_descriptor[
            "difference_vector_relative_l2_error"
        ],
        "minimum_difference_vector_cosine": cpu_descriptor[
            "difference_vector_cosine"
        ],
        "maximum_separation_magnitude_error": abs(
            cpu_descriptor["float32_over_reference_separation"] - 1.0
        ),
    }
    candidates = {}
    for frontier, result in gpu_result["prototype_frontiers"].items():
        descriptor = result["stages"]["descriptor_final"]
        score_error = result["production_weighted_score"][
            "similarity_score_absolute_error"
        ]
        checks = {
            "similarity_score": score_error
            <= thresholds["maximum_similarity_score_absolute_error"],
            "difference_vector_error": descriptor[
                "difference_vector_relative_l2_error"
            ]
            <= thresholds["maximum_difference_vector_relative_l2_error"],
            "difference_vector_cosine": descriptor["difference_vector_cosine"]
            >= thresholds["minimum_difference_vector_cosine"],
            "separation_magnitude": abs(
                descriptor["float32_over_reference_separation"] - 1.0
            )
            <= thresholds["maximum_separation_magnitude_error"],
        }
        candidates[frontier] = {"passes": all(checks.values()), "checks": checks}
    gpu_result["cpu_float32_accuracy_gate"] = {
        "thresholds": thresholds,
        "candidates": candidates,
    }


def _run_isolated(max_order: int, tmp_path: Path) -> dict[str, object]:
    backend = os.getenv("ZMPY3D_FLOAT32_REGRESSION_BACKEND", "cpu").lower()
    if backend not in ("cpu", "gpu"):
        raise ValueError("ZMPY3D_FLOAT32_REGRESSION_BACKEND must be 'cpu' or 'gpu'")
    result = _compare(max_order, backend, tmp_path)
    if max_order == 20 and backend == "gpu":
        cpu_directory = tmp_path / "cpu_baseline"
        cpu_directory.mkdir()
        cpu_result = _compare(max_order, "cpu", cpu_directory)
        _apply_cpu_accuracy_gate(result, cpu_result)
        assert all(
            candidate["passes"]
            for candidate in result["cpu_float32_accuracy_gate"][
                "candidates"
            ].values()
        )
    output_dir = os.getenv("ZMPY3D_FLOAT32_ACCURACY_OUTPUT")
    if output_dir:
        destination = Path(output_dir).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / f"float32_structure_accuracy_{backend}_order{max_order}.json"
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def test_float32_order6_preserves_structure_separation(tmp_path: Path) -> None:
    result = _run_isolated(6, tmp_path)
    assert result["schema_version"] == SCHEMA_VERSION


@pytest.mark.slow
def test_float32_order20_preserves_structure_separation(tmp_path: Path) -> None:
    result = _run_isolated(20, tmp_path)
    assert "production_weighted_score" in result


if __name__ == "__main__":
    if len(sys.argv) not in (6, 7) or sys.argv[1] != "--worker":
        raise SystemExit(
            "usage: test_float32_structure_accuracy.py "
            "--worker ORDER BACKEND {x64|float32} OUTPUT"
        )
    _run_worker(
        int(sys.argv[2]),
        sys.argv[3],
        sys.argv[4] == "x64",
        Path(sys.argv[5]),
        Path(sys.argv[6]) if len(sys.argv) == 7 else None,
    )
