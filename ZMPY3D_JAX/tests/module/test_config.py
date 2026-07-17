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
roots = z.eigen_root(np.array([1.0, -3.0, 2.0]))
assert roots.dtype.name == {complex_name!r}
rng = np.random.default_rng(7)
raw = rng.normal(size=(7, 7, 13)) + 1j * rng.normal(size=(7, 7, 13))
single = z.calculate_ab_rotation(raw, 2)
all_orders = z.calculate_ab_rotation_all(raw, 2)
assert single.dtype.name == {complex_name!r}
assert all(array.dtype.name == {complex_name!r} for array in all_orders)
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
