import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.dont_write_bytecode = True
import ZMPY3D_JAX as z

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "externals/ZMPY3D"))

from ZMPY3D.lib.calculate_bbox_moment06 import calculate_bbox_moment06 as upstream_bbox
from ZMPY3D.lib.calculate_bbox_moment_2_zm05 import (
    calculate_bbox_moment_2_zm05 as upstream_bbox_to_zm,
)
from ZMPY3D.lib.fill_voxel_by_weight_density04 import (
    fill_voxel_by_weight_density04 as upstream_fill_voxel,
)
from ZMPY3D.lib.get_3dzd_121_descriptor02 import (
    get_3dzd_121_descriptor02 as upstream_descriptor,
)


@pytest.mark.filterwarnings("ignore:numpy.fix is deprecated:DeprecationWarning")
def test_core_pipeline_matches_upstream():
    params = z.get_global_parameter()
    boxes = z.get_residue_gaussian_density_cache(params)[1.0]
    xyz = np.array([[0.0, 0.0, 0.0], [2.5, 1.0, -0.5], [-1.0, 2.0, 1.5]])
    residues = ["ALA", "GLY", "VAL"]

    actual_voxel, actual_corner = z.fill_voxel_by_weight_density(
        xyz, residues, params["residue_weight_map"], 1.0, boxes
    )
    expected_voxel, expected_corner = upstream_fill_voxel(
        xyz, residues, params["residue_weight_map"], 1.0, boxes
    )
    np.testing.assert_allclose(actual_voxel, expected_voxel, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_corner, expected_corner, rtol=0, atol=1e-7)

    samples = {
        "X_sample": np.linspace(-1, 1, actual_voxel.shape[0] + 1),
        "Y_sample": np.linspace(-1, 1, actual_voxel.shape[1] + 1),
        "Z_sample": np.linspace(-1, 1, actual_voxel.shape[2] + 1),
    }
    actual_mass, actual_center, actual_bbox = z.calculate_bbox_moment(actual_voxel, 6, samples)
    expected_mass, expected_center, expected_bbox = upstream_bbox(expected_voxel, 6, samples)
    np.testing.assert_allclose(actual_mass, expected_mass, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(actual_center, expected_center, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(actual_bbox, expected_bbox, rtol=3e-5, atol=3e-6)

    with open(z.__path__[0] + "/cache_data/LogG_CLMCache_MaxOrder06.pkl", "rb") as handle:
        cache = pickle.load(handle)
    conversion_args = (
        6,
        cache["GCache_complex"],
        cache["GCache_pqr_linear"],
        cache["GCache_complex_index"],
        cache["CLMCache3D"],
    )
    actual_scaled, actual_raw = z.calculate_bbox_moment_2_zm(*conversion_args, actual_bbox)
    expected_scaled, expected_raw = upstream_bbox_to_zm(*conversion_args, expected_bbox)
    np.testing.assert_allclose(actual_raw, expected_raw, rtol=5e-5, atol=5e-6, equal_nan=True)
    np.testing.assert_allclose(
        actual_scaled, expected_scaled, rtol=5e-5, atol=5e-6, equal_nan=True
    )

    actual_descriptor = z.get_3dzd_121_descriptor(actual_scaled)
    expected_descriptor = upstream_descriptor(expected_scaled.copy())
    np.testing.assert_allclose(
        actual_descriptor, expected_descriptor, rtol=5e-5, atol=5e-6, equal_nan=True
    )
