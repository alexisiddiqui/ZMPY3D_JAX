import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("enable_x64", "float_name", "complex_name"),
    [(True, "float64", "complex128"), (False, "float32", "complex64")],
)
def test_runtime_dtype_configuration(enable_x64, float_name, complex_name):
    code = f"""
import numpy as np
import ZMPY3D_JAX as z
z.configure_for_scientific_computing(enable_x64={enable_x64!r})
samples = z.get_bbox_moment_xyz_sample([1, 1, 1], 2, (2, 2, 2))
assert z.FLOAT_DTYPE.__name__ == {float_name!r}
assert z.COMPLEX_DTYPE.__name__ == {complex_name!r}
assert all(array.dtype.name == {float_name!r} for array in samples.values())
voxel = np.zeros((3, 3, 3), dtype=float)
voxel[1, 1, 1] = 1
bbox_mass, bbox_center, bbox_moments = z.calculate_bbox_moment(
    np.ones((2, 2, 2)), 2, samples
)
assert bbox_mass.dtype.name == {float_name!r}
assert bbox_center.dtype.name == {float_name!r}
assert bbox_moments.dtype.name == {float_name!r}
bbox_to_zm_cache = z.prepare_bbox_to_zm_cache(
    1,
    np.ones(1, dtype=complex),
    np.ones(1, dtype=np.int64),
    np.ones(1, dtype=np.int64),
    np.ones((2, 2, 2), dtype=complex),
)
cached_scaled, cached_raw = z.calculate_bbox_moment_2_zm_cached(
    np.ones((2, 2, 2)), bbox_to_zm_cache
)
assert bbox_to_zm_cache.g_coefficients.dtype.name == {complex_name!r}
assert bbox_to_zm_cache.clm.dtype.name == {complex_name!r}
assert bbox_to_zm_cache.pqr_indices.dtype.name == "int32"
assert bbox_to_zm_cache.output_indices.dtype.name == "int32"
assert cached_scaled.dtype.name == {complex_name!r}
assert cached_raw.dtype.name == {complex_name!r}
average, maximum, fused_samples = z.calculate_molecular_radius_and_bbox_samples(
    voxel, np.array([0.5, 0.5, 0.5]), 1.0, 1.8
)
assert average.dtype.name == {float_name!r}
assert maximum.dtype.name == {float_name!r}
assert all(array.dtype.name == {float_name!r} for array in fused_samples.values())
roots = z.eigen_root(np.array([1.0, -3.0, 2.0]))
assert roots.dtype.name == {complex_name!r}
rng = np.random.default_rng(7)
raw = rng.normal(size=(7, 7, 13)) + 1j * rng.normal(size=(7, 7, 13))
single = z.calculate_ab_rotation(raw, 2)
all_orders = z.calculate_ab_rotation_all(raw, 2)
fixed_single = z.calculate_ab_rotation_candidates(raw, 2)
fixed_all = z.calculate_ab_rotation_all_candidates(raw, 2)
assert single.dtype.name == {complex_name!r}
assert all(array.dtype.name == {complex_name!r} for array in all_orders)
assert fixed_single.pairs.dtype.name == {complex_name!r}
assert fixed_all.pairs.dtype.name == {complex_name!r}
assert fixed_single.is_valid.dtype.name == "bool"
assert fixed_all.is_valid.dtype.name == "bool"
params = z.get_global_parameter()
residue_boxes = z.get_residue_gaussian_density_cache(params)
assert all(
    isinstance(box, np.ndarray) and box.dtype.name == {float_name!r}
    for boxes in residue_boxes.values()
    for box in boxes.values()
)
rotation_cache = z.prepare_zm_rotation_cache(
    np.ones((2, 2)), 1, np.ones(4),
    *[np.array([0], dtype=np.int64) for _ in range(7)],
)
assert rotation_cache.binomial.dtype.name == {float_name!r}
assert rotation_cache.clm.dtype.name == {float_name!r}
assert all(array.dtype.name == "int32" for array in rotation_cache[3:])
"""
    subprocess.run([sys.executable, "-c", code], check=True)
