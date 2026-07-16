import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("enable_x64", "float_name", "complex_name"),
    [(True, "float64", "complex128"), (False, "float32", "complex64")],
)
def test_runtime_dtype_configuration(enable_x64, float_name, complex_name):
    code = f"""
import ZMPY3D_JAX as z
z.configure_for_scientific_computing(enable_x64={enable_x64!r})
samples = z.get_bbox_moment_xyz_sample([1, 1, 1], 2, (2, 2, 2))
assert z.FLOAT_DTYPE.__name__ == {float_name!r}
assert z.COMPLEX_DTYPE.__name__ == {complex_name!r}
assert all(array.dtype.name == {float_name!r} for array in samples.values())
"""
    subprocess.run([sys.executable, "-c", code], check=True)
