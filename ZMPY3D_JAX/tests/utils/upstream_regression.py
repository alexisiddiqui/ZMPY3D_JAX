"""Shared helpers for end-to-end parity and performance checks against ZMPY3D."""

from __future__ import annotations

import importlib
import pickle
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

import ZMPY3D_JAX as z


REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = REPO_ROOT / "externals" / "ZMPY3D"

if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))


@dataclass(frozen=True)
class RegressionInput:
    """One deterministic protein CA-trace regression input."""

    name: str
    xyz: np.ndarray
    residues: tuple[str, ...]
    grid_width: float = 1.0
    max_order: int = 6

    def with_order(self, max_order: int) -> "RegressionInput":
        return replace(self, name=f"{self.name}-order{max_order}", max_order=max_order)


@dataclass(frozen=True)
class PipelineContext:
    """Materialized implementation-specific setup for one regression input."""

    implementation: str
    case: RegressionInput
    cache: dict[str, Any]
    params: dict[str, Any]
    residue_boxes: dict[float, Any]
    functions: dict[str, Callable[..., Any]]
    rotation_cache: Any | None = None
    bbox_to_zm_cache: Any | None = None


def block_tree(value: Any) -> None:
    """Wait for every asynchronous JAX leaf in a result tree."""
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    elif isinstance(value, dict):
        for item in value.values():
            block_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            block_tree(item)


def canonicalize_ab_pairs(values: np.ndarray) -> np.ndarray:
    """Sort AB pairs deterministically without changing their values."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] == 0:
        return values
    order = np.lexsort(
        (values[:, 1].imag, values[:, 1].real, values[:, 0].imag, values[:, 0].real)
    )
    return values[order]


def synthetic_inputs() -> tuple[RegressionInput, ...]:
    """Return mixed-residue cases spanning the supported committed grid widths."""
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.5, 1.0, -0.5],
            [-1.0, 2.0, 1.5],
            [3.2, -1.7, 2.1],
            [-2.4, -0.8, 0.7],
            [1.1, 3.5, -2.0],
            [4.0, 2.2, 2.8],
            [-3.1, 1.4, -1.6],
        ],
        dtype=np.float64,
    )
    residues = ("ALA", "GLY", "VAL", "LYS", "ASP", "TRP", "CYS", "PRO")
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return (
        RegressionInput("mixed-base-gw1", xyz, residues, grid_width=1.0),
        RegressionInput(
            "mixed-translated-gw05",
            xyz + np.array([17.25, -8.5, 3.75]),
            residues,
            grid_width=0.5,
        ),
        RegressionInput(
            "mixed-rotated-gw025",
            xyz @ rotation.T,
            residues,
            grid_width=0.25,
        ),
    )


def pdb_input(path: str | Path, *, max_order: int = 6) -> RegressionInput:
    """Parse a real PDB fixture with the JAX parser for pipeline input."""
    path = Path(path)
    xyz, residues = z.get_pdb_xyz_ca(str(path))
    return RegressionInput(
        name=f"{path.stem}-gw1-order{max_order}",
        xyz=np.asarray(xyz),
        residues=tuple(residues),
        grid_width=1.0,
        max_order=max_order,
    )


@lru_cache(maxsize=1)
def upstream_functions() -> dict[str, Any]:
    """Load the original NumPy functions without importing its CLI wrappers."""
    names = {
        "get_global_parameter": ("ZMPY3D.lib.get_global_parameter02", "get_global_parameter02"),
        "get_residue_cache": (
            "ZMPY3D.lib.get_residue_gaussian_density_cache02",
            "get_residue_gaussian_density_cache02",
        ),
        "parse_pdb": ("ZMPY3D.lib.get_pdb_xyz_ca02", "get_pdb_xyz_ca02"),
        "fill_voxel": (
            "ZMPY3D.lib.fill_voxel_by_weight_density04",
            "fill_voxel_by_weight_density04",
        ),
        "bbox": ("ZMPY3D.lib.calculate_bbox_moment06", "calculate_bbox_moment06"),
        "radius": (
            "ZMPY3D.lib.calculate_molecular_radius03",
            "calculate_molecular_radius03",
        ),
        "xyz_sample": (
            "ZMPY3D.lib.get_bbox_moment_xyz_sample01",
            "get_bbox_moment_xyz_sample01",
        ),
        "bbox_to_zm": (
            "ZMPY3D.lib.calculate_bbox_moment_2_zm05",
            "calculate_bbox_moment_2_zm05",
        ),
        "descriptor": (
            "ZMPY3D.lib.get_3dzd_121_descriptor02",
            "get_3dzd_121_descriptor02",
        ),
        "ab": ("ZMPY3D.lib.calculate_ab_rotation_02", "calculate_ab_rotation_02"),
        "ab_all": (
            "ZMPY3D.lib.calculate_ab_rotation_02_all",
            "calculate_ab_rotation_02_all",
        ),
        "rotate": (
            "ZMPY3D.lib.calculate_zm_by_ab_rotation01",
            "calculate_zm_by_ab_rotation01",
        ),
    }
    loaded = {}
    for key, (module_name, function_name) in names.items():
        loaded[key] = getattr(importlib.import_module(module_name), function_name)
    return loaded


@lru_cache(maxsize=2)
def load_cache(max_order: int) -> dict[str, Any]:
    """Load shared, byte-identical JAX/upstream caches for one maximum order."""
    cache_dir = Path(z.__file__).parent / "cache_data"
    with (cache_dir / f"LogG_CLMCache_MaxOrder{max_order:02d}.pkl").open("rb") as handle:
        cache = pickle.load(handle)
    with (cache_dir / "BinomialCache.pkl").open("rb") as handle:
        binomial = pickle.load(handle)["BinomialCache"]

    rotation_index = cache["RotationIndex"]
    return {
        "BinomialCache": binomial,
        "GCache_complex": cache["GCache_complex"],
        "GCache_pqr_linear": cache["GCache_pqr_linear"],
        "GCache_complex_index": cache["GCache_complex_index"],
        "CLMCache3D": cache["CLMCache3D"],
        "CLMCache": cache["CLMCache"],
        "s_id": np.squeeze(rotation_index["s_id"][0, 0]) - 1,
        "n": np.squeeze(rotation_index["n"][0, 0]),
        "l": np.squeeze(rotation_index["l"][0, 0]),
        "m": np.squeeze(rotation_index["m"][0, 0]),
        "mu": np.squeeze(rotation_index["mu"][0, 0]),
        "k": np.squeeze(rotation_index["k"][0, 0]),
        "IsNLM_Value": np.squeeze(rotation_index["IsNLM_Value"][0, 0]) - 1,
    }


def build_jax_setup() -> tuple[dict[str, Any], dict[float, Any]]:
    """Build uncached JAX parameters and residue-density boxes."""
    params = z.get_global_parameter()
    return params, z.get_residue_gaussian_density_cache(params)


def build_upstream_setup() -> tuple[dict[str, Any], dict[float, Any]]:
    """Build uncached upstream parameters and residue-density boxes."""
    functions = upstream_functions()
    params = functions["get_global_parameter"]()
    return params, functions["get_residue_cache"](params)


@lru_cache(maxsize=1)
def jax_setup() -> tuple[dict[str, Any], dict[float, Any]]:
    return build_jax_setup()


@lru_cache(maxsize=1)
def upstream_setup() -> tuple[dict[str, Any], dict[float, Any]]:
    return build_upstream_setup()


def prepare_pipeline_context(
    implementation: str,
    case: RegressionInput,
    *,
    cache: dict[str, Any] | None = None,
    setup: tuple[dict[str, Any], dict[float, Any]] | None = None,
    rotation_cache: Any | None = None,
    bbox_to_zm_cache: Any | None = None,
) -> PipelineContext:
    """Resolve functions and setup before a pipeline is timed or executed."""
    if implementation not in {"jax", "upstream"}:
        raise ValueError(f"Unknown implementation: {implementation}")

    cache = load_cache(case.max_order) if cache is None else cache
    if implementation == "jax":
        params, residue_boxes = jax_setup() if setup is None else setup
        functions = {
            "fill_voxel": z.fill_voxel_by_weight_density,
            "bbox": z.calculate_bbox_moment,
            "radius": z.calculate_molecular_radius,
            "radius_and_sphere": z.calculate_molecular_radius_and_bbox_samples,
            "xyz_sample": z.get_bbox_moment_xyz_sample,
            "bbox_to_zm": z.calculate_bbox_moment_2_zm,
            "bbox_to_zm_cached": z.calculate_bbox_moment_2_zm_cached,
            "descriptor": z.get_3dzd_121_descriptor,
            "ab": z.calculate_ab_rotation,
            "ab_all": z.calculate_ab_rotation_all,
            "ab_fixed": z.calculate_ab_rotation_candidates,
            "ab_all_fixed": z.calculate_ab_rotation_all_candidates,
            "rotate": z.calculate_zm_by_ab_rotation,
            "rotate_batch": z.calculate_zm_by_ab_rotation_batch,
        }
        if rotation_cache is None:
            rotation_cache = prepare_jax_rotation_cache(case, cache)
        if bbox_to_zm_cache is None:
            bbox_to_zm_cache = prepare_jax_bbox_to_zm_cache(case, cache)
    else:
        params, residue_boxes = upstream_setup() if setup is None else setup
        upstream = upstream_functions()
        functions = {key: upstream[key] for key in (
            "fill_voxel",
            "bbox",
            "radius",
            "xyz_sample",
            "bbox_to_zm",
            "descriptor",
            "ab",
            "ab_all",
            "rotate",
        )}
        rotation_cache = None
        bbox_to_zm_cache = None

    return PipelineContext(
        implementation=implementation,
        case=case,
        cache=cache,
        params=params,
        residue_boxes=residue_boxes,
        functions=functions,
        rotation_cache=rotation_cache,
        bbox_to_zm_cache=bbox_to_zm_cache,
    )


def prepare_jax_rotation_cache(
    case: RegressionInput, cache: dict[str, Any]
) -> z.ZMRotationCache:
    """Materialize the JAX-only rotation constants for a prepared pipeline."""
    return z.prepare_zm_rotation_cache(
        cache["BinomialCache"],
        case.max_order,
        cache["CLMCache"],
        cache["s_id"],
        cache["n"],
        cache["l"],
        cache["m"],
        cache["mu"],
        cache["k"],
        cache["IsNLM_Value"],
    )


def prepare_jax_bbox_to_zm_cache(
    case: RegressionInput, cache: dict[str, Any]
) -> z.BBoxToZMCache:
    """Materialize the JAX-only bbox-to-ZM constants for a prepared pipeline."""
    return z.prepare_bbox_to_zm_cache(
        case.max_order,
        cache["GCache_complex"],
        cache["GCache_pqr_linear"],
        cache["GCache_complex_index"],
        cache["CLMCache3D"],
    )


def _rotation_args(raw: Any, ab_pairs: np.ndarray, max_order: int, cache: dict[str, Any]):
    return (
        raw,
        cache["BinomialCache"],
        ab_pairs,
        max_order,
        cache["CLMCache"],
        cache["s_id"],
        cache["n"],
        cache["l"],
        cache["m"],
        cache["mu"],
        cache["k"],
        cache["IsNLM_Value"],
    )


def _canonical_candidate_batch(candidates: dict[int, list[np.ndarray]]) -> np.ndarray:
    arrays = [
        canonicalize_ab_pairs(item)
        for order in sorted(candidates)
        for item in candidates[order]
        if np.asarray(item).size
    ]
    return np.vstack(arrays) if arrays else np.empty((0, 2), dtype=np.complex128)


StageExecutor = Callable[[str, Callable[[], Any]], Any]


def _execute_directly(_name: str, function: Callable[[], Any]) -> Any:
    return function()


def run_prepared_pipeline(
    context: PipelineContext,
    *,
    normalization_orders: Iterable[int] = range(2, 6),
    all_candidates: bool = True,
    stage_executor: StageExecutor | None = None,
) -> dict[str, Any]:
    """Run a prepared pipeline, optionally routing work through named stage timers."""
    case = context.case
    cache = context.cache
    params = context.params
    residue_boxes = context.residue_boxes
    functions = context.functions
    execute = _execute_directly if stage_executor is None else stage_executor

    voxel, corner = execute(
        "voxelization",
        lambda: functions["fill_voxel"](
            np.asarray(case.xyz),
            list(case.residues),
            params["residue_weight_map"],
            case.grid_width,
            residue_boxes[case.grid_width],
        ),
    )
    samples = {
        "X_sample": np.arange(voxel.shape[0] + 1, dtype=float),
        "Y_sample": np.arange(voxel.shape[1] + 1, dtype=float),
        "Z_sample": np.arange(voxel.shape[2] + 1, dtype=float),
    }
    mass, center, bbox_order1 = execute(
        "bbox_order1", lambda: functions["bbox"](voxel, 1, samples)
    )

    def radius_and_sphere():
        if "radius_and_sphere" in functions:
            return functions["radius_and_sphere"](
                voxel, center, mass, params["default_radius_multiplier"]
            )
        average_radius, max_radius = functions["radius"](
            voxel, center, mass, params["default_radius_multiplier"]
        )
        sphere = functions["xyz_sample"](center, average_radius, voxel.shape)
        return average_radius, max_radius, sphere

    average_radius, max_radius, sphere = execute(
        "radius_and_sphere", radius_and_sphere
    )
    mass_n, center_n, bbox_order_n = execute(
        "bbox_max_order",
        lambda: functions["bbox"](voxel, case.max_order, sphere),
    )
    def bbox_to_zm():
        if context.bbox_to_zm_cache is not None:
            return functions["bbox_to_zm_cached"](
                bbox_order_n, context.bbox_to_zm_cache
            )
        return functions["bbox_to_zm"](
            case.max_order,
            cache["GCache_complex"],
            cache["GCache_pqr_linear"],
            cache["GCache_complex_index"],
            cache["CLMCache3D"],
            bbox_order_n,
        )

    scaled, raw = execute("bbox_to_zm", bbox_to_zm)
    descriptor_value = execute(
        "descriptor", lambda: functions["descriptor"](np.asarray(scaled).copy())
    )

    def build_candidates():
        candidates: dict[int, list[np.ndarray]] = {}
        for order in normalization_orders:
            if "ab_fixed" in functions:
                fixed_function = (
                    functions["ab_all_fixed"] if all_candidates else functions["ab_fixed"]
                )
                fixed = fixed_function(raw, order)
                pairs = np.asarray(fixed.pairs)
                is_valid = np.asarray(fixed.is_valid)
                candidates[order] = (
                    [pairs[index][is_valid[index]] for index in range(pairs.shape[0])]
                    if all_candidates
                    else [pairs[is_valid]]
                )
            else:
                ab_function = functions["ab_all"] if all_candidates else functions["ab"]
                value = ab_function(raw, order)
                candidates[order] = (
                    list(value) if all_candidates else [np.asarray(value)]
                )
        return candidates, _canonical_candidate_batch(candidates)

    candidates, candidate_batch = execute("ab_candidates", build_candidates)
    def rotate():
        if context.rotation_cache is None:
            return functions["rotate"](
                *_rotation_args(raw, candidate_batch, case.max_order, cache)
            )
        return functions["rotate_batch"](
            raw, candidate_batch, context.rotation_cache
        )

    rotated = execute("zm_rotation", rotate)

    result = {
        "voxel": voxel,
        "corner": corner,
        "mass": mass,
        "center": center,
        "bbox_order1": bbox_order1,
        "average_radius": average_radius,
        "max_radius": max_radius,
        "sphere": sphere,
        "mass_n": mass_n,
        "center_n": center_n,
        "bbox_order_n": bbox_order_n,
        "scaled": scaled,
        "raw": raw,
        "descriptor": descriptor_value,
        "candidates": candidates,
        "rotated": rotated,
    }
    block_tree(result)
    return result


def run_pipeline(
    implementation: str,
    case: RegressionInput,
    *,
    normalization_orders: Iterable[int] = range(2, 6),
    all_candidates: bool = True,
) -> dict[str, Any]:
    """Run exactly one implementation and return comparable stage outputs."""
    context = prepare_pipeline_context(implementation, case)
    return run_prepared_pipeline(
        context,
        normalization_orders=normalization_orders,
        all_candidates=all_candidates,
    )
