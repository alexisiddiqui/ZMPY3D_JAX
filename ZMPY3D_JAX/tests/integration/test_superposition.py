import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import ZMPY3D_JAX as z
from ZMPY3D_JAX.tests.utils.pdbFixture import pdb_files, output_dir


@pytest.fixture
def global_params():
    """Get global parameters for ZMPY3D."""
    return z.get_global_parameter()


@pytest.fixture
def residue_box(global_params):
    """Get residue Gaussian density cache."""
    return z.get_residue_gaussian_density_cache(global_params)


@pytest.fixture
def cache_data():
    """Load precalculated cache data."""
    max_order = 6

    cache_dir = Path(z.__file__).parent / "cache_data"
    binomial_path = cache_dir / "BinomialCache.pkl"
    log_cache_path = cache_dir / f"LogG_CLMCache_MaxOrder{max_order:02d}.pkl"

    with open(binomial_path, "rb") as file:
        binomial_cache_pkl = pickle.load(file)

    with open(log_cache_path, "rb") as file:
        cache_pkl = pickle.load(file)

    rotation_index = cache_pkl["RotationIndex"]

    return {
        "BinomialCache": binomial_cache_pkl["BinomialCache"],
        "GCache_pqr_linear": cache_pkl["GCache_pqr_linear"],
        "GCache_complex": cache_pkl["GCache_complex"],
        "GCache_complex_index": cache_pkl["GCache_complex_index"],
        "CLMCache3D": cache_pkl["CLMCache3D"],
        "CLMCache": cache_pkl["CLMCache"],
        "s_id": np.squeeze(rotation_index["s_id"][0, 0]) - 1,
        "n": np.squeeze(rotation_index["n"][0, 0]),
        "l": np.squeeze(rotation_index["l"][0, 0]),
        "m": np.squeeze(rotation_index["m"][0, 0]),
        "mu": np.squeeze(rotation_index["mu"][0, 0]),
        "k": np.squeeze(rotation_index["k"][0, 0]),
        "IsNLM_Value": np.squeeze(rotation_index["IsNLM_Value"][0, 0]) - 1,
    }


def one_time_conversion(voxel3d, corner, grid_width, cache_data, global_params, max_order=6):
    """
    Convert voxel data to Zernike moments with all rotations.
    """
    dimension_bbox_scaled = voxel3d.shape

    xyz_sample_struct = {
        "X_sample": np.arange(dimension_bbox_scaled[0] + 1),
        "Y_sample": np.arange(dimension_bbox_scaled[1] + 1),
        "Z_sample": np.arange(dimension_bbox_scaled[2] + 1),
    }

    volume_mass, center, _ = z.calculate_bbox_moment(voxel3d, 1, xyz_sample_struct)

    average_voxel_dist2center, _ = z.calculate_molecular_radius(
        voxel3d, center, volume_mass, global_params["default_radius_multiplier"]
    )

    center_scaled = center * grid_width + corner

    sphere_xyz_sample_struct = z.get_bbox_moment_xyz_sample(
        center, average_voxel_dist2center, dimension_bbox_scaled
    )

    _, _, sphere_bbox_moment = z.calculate_bbox_moment(voxel3d, max_order, sphere_xyz_sample_struct)

    _, z_moment_raw = z.calculate_bbox_moment_2_zm(
        max_order,
        cache_data["GCache_complex"],
        cache_data["GCache_pqr_linear"],
        cache_data["GCache_complex_index"],
        cache_data["CLMCache3D"],
        sphere_bbox_moment,
    )

    ab_list_2 = z.calculate_ab_rotation_all(z_moment_raw, 2)
    ab_list_3 = z.calculate_ab_rotation_all(z_moment_raw, 3)
    ab_list_4 = z.calculate_ab_rotation_all(z_moment_raw, 4)
    ab_list_5 = z.calculate_ab_rotation_all(z_moment_raw, 5)
    ab_list_6 = z.calculate_ab_rotation_all(z_moment_raw, 6)

    ab_list_all = np.vstack(ab_list_2 + ab_list_3 + ab_list_4 + ab_list_5 + ab_list_6)

    zm_list_all = z.calculate_zm_by_ab_rotation(
        z_moment_raw,
        cache_data["BinomialCache"],
        ab_list_all,
        max_order,
        cache_data["CLMCache"],
        cache_data["s_id"],
        cache_data["n"],
        cache_data["l"],
        cache_data["m"],
        cache_data["mu"],
        cache_data["k"],
        cache_data["IsNLM_Value"],
    )

    zm_list_all = np.stack(zm_list_all, axis=3)
    zm_list_all = np.transpose(zm_list_all, (2, 1, 0, 3))
    zm_list_all = zm_list_all[~np.isnan(zm_list_all)]
    zm_list_all = np.reshape(zm_list_all, (np.int64(zm_list_all.size / 96), 96))

    return center_scaled, ab_list_all, zm_list_all


class TestSuperposition:
    """Integration tests for molecular superposition using Zernike moments."""

    def test_voxel_conversion(self, pdb_files, global_params, residue_box):
        """Test conversion of PDB structures to voxel grids."""
        grid_width = 1.0

        xyz_a, aa_name_list_a = z.get_pdb_xyz_ca(pdb_files["6NT5"])
        voxel3d_a, corner_a = z.fill_voxel_by_weight_density(
            xyz_a,
            aa_name_list_a,
            global_params["residue_weight_map"],
            grid_width,
            residue_box[grid_width],
        )

        xyz_b, aa_name_list_b = z.get_pdb_xyz_ca(pdb_files["6NT6"])
        voxel3d_b, corner_b = z.fill_voxel_by_weight_density(
            xyz_b,
            aa_name_list_b,
            global_params["residue_weight_map"],
            grid_width,
            residue_box[grid_width],
        )

        # Assert voxels were created
        assert voxel3d_a.shape[0] > 0
        assert voxel3d_a.shape[1] > 0
        assert voxel3d_a.shape[2] > 0
        assert voxel3d_b.shape[0] > 0
        assert voxel3d_b.shape[1] > 0
        assert voxel3d_b.shape[2] > 0

        # Assert voxels contain non-zero values
        assert np.sum(voxel3d_a) > 0
        assert np.sum(voxel3d_b) > 0

    def test_zernike_moment_calculation(self, pdb_files, global_params, residue_box, cache_data):
        """Test calculation of Zernike moments with rotations."""
        grid_width = 1.0

        xyz_a, aa_name_list_a = z.get_pdb_xyz_ca(pdb_files["6NT5"])
        voxel3d_a, corner_a = z.fill_voxel_by_weight_density(
            xyz_a,
            aa_name_list_a,
            global_params["residue_weight_map"],
            grid_width,
            residue_box[grid_width],
        )

        center_scaled_a, ab_list_a, zm_list_a = one_time_conversion(
            voxel3d_a, corner_a, grid_width, cache_data, global_params
        )

        # Assert center is calculated
        assert center_scaled_a.shape == (3,)

        # Assert AB list has expected shape (96 rotations for order 6)
        assert ab_list_a.shape[1] == 2  # a and b coefficients

        # Assert ZM list has expected dimensions (96 features per rotation)
        assert zm_list_a.shape[1] == 96

        # Assert values are complex
        assert np.iscomplexobj(zm_list_a)

    def test_transformation_matrix_calculation(
        self, pdb_files, global_params, residue_box, cache_data, output_dir
    ):
        """Test calculation of transformation matrix for superposition."""
        grid_width = 1.0

        # Process structure A
        xyz_a, aa_name_list_a = z.get_pdb_xyz_ca(pdb_files["6NT5"])
        voxel3d_a, corner_a = z.fill_voxel_by_weight_density(
            xyz_a,
            aa_name_list_a,
            global_params["residue_weight_map"],
            grid_width,
            residue_box[grid_width],
        )
        center_scaled_a, ab_list_a, zm_list_a = one_time_conversion(
            voxel3d_a, corner_a, grid_width, cache_data, global_params
        )

        # Process structure B
        xyz_b, aa_name_list_b = z.get_pdb_xyz_ca(pdb_files["6NT6"])
        voxel3d_b, corner_b = z.fill_voxel_by_weight_density(
            xyz_b,
            aa_name_list_b,
            global_params["residue_weight_map"],
            grid_width,
            residue_box[grid_width],
        )
        center_scaled_b, ab_list_b, zm_list_b = one_time_conversion(
            voxel3d_b, corner_b, grid_width, cache_data, global_params
        )

        # Calculate transformation matrix
        m = np.abs(zm_list_a.conj().T @ zm_list_b)
        max_value_index = np.where(m == np.max(m))
        i, j = max_value_index[0][0], max_value_index[1][0]

        rot_m_a = z.get_transform_matrix_from_ab_list(
            ab_list_a[i, 0], ab_list_a[i, 1], center_scaled_a
        )
        rot_m_b = z.get_transform_matrix_from_ab_list(
            ab_list_b[j, 0], ab_list_b[j, 1], center_scaled_b
        )
        target_rot_m = np.linalg.solve(rot_m_b, rot_m_a)

        # Assertions
        assert target_rot_m.shape == (4, 4)

        # Transformation matrix should preserve the homogeneous coordinate
        assert np.isclose(target_rot_m[3, 3], 1.0)

        # Bottom row should be [0, 0, 0, 1]
        assert np.allclose(target_rot_m[3, :3], 0.0)

        # Save output
        np.save(output_dir / "transformation_matrix.npy", target_rot_m)

        # Save similarity matrix
        np.save(output_dir / "similarity_matrix.npy", m)

    def test_pdb_transformation(
        self, pdb_files, global_params, residue_box, cache_data, output_dir
    ):
        """Test application of transformation matrix to PDB file."""
        grid_width = 1.0

        # Calculate transformation matrix (reusing logic)
        xyz_a, aa_name_list_a = z.get_pdb_xyz_ca(pdb_files["6NT5"])
        voxel3d_a, corner_a = z.fill_voxel_by_weight_density(
            xyz_a,
            aa_name_list_a,
            global_params["residue_weight_map"],
            grid_width,
            residue_box[grid_width],
        )
        center_scaled_a, ab_list_a, zm_list_a = one_time_conversion(
            voxel3d_a, corner_a, grid_width, cache_data, global_params
        )

        xyz_b, aa_name_list_b = z.get_pdb_xyz_ca(pdb_files["6NT6"])
        voxel3d_b, corner_b = z.fill_voxel_by_weight_density(
            xyz_b,
            aa_name_list_b,
            global_params["residue_weight_map"],
            grid_width,
            residue_box[grid_width],
        )
        center_scaled_b, ab_list_b, zm_list_b = one_time_conversion(
            voxel3d_b, corner_b, grid_width, cache_data, global_params
        )

        m = np.abs(zm_list_a.conj().T @ zm_list_b)
        max_value_index = np.where(m == np.max(m))
        i, j = max_value_index[0][0], max_value_index[1][0]

        rot_m_a = z.get_transform_matrix_from_ab_list(
            ab_list_a[i, 0], ab_list_a[i, 1], center_scaled_a
        )
        rot_m_b = z.get_transform_matrix_from_ab_list(
            ab_list_b[j, 0], ab_list_b[j, 1], center_scaled_b
        )
        target_rot_m = np.linalg.solve(rot_m_b, rot_m_a)

        # Apply transformation
        output_pdb = output_dir / "6NT5_transformed.pdb"
        z.set_pdb_xyz_rot(pdb_files["6NT5"], target_rot_m, str(output_pdb))

        # Assert output file was created
        assert output_pdb.exists()

        # Read transformed coordinates
        xyz_transformed, _ = z.get_pdb_xyz_ca(str(output_pdb))

        # Assert transformation was applied (coordinates should be different)
        assert not np.allclose(xyz_a, xyz_transformed)

        # Assert number of atoms is preserved
        assert xyz_a.shape[0] == xyz_transformed.shape[0]

        # Save transformed coordinates for inspection
        np.save(output_dir / "original_coords.npy", xyz_a)
        np.save(output_dir / "transformed_coords.npy", xyz_transformed)

    def test_superposition_quality(
        self, pdb_files, global_params, residue_box, cache_data, output_dir
    ):
        """Test the quality of superposition by measuring alignment."""
        grid_width = 1.0

        # Full superposition workflow
        xyz_a, aa_name_list_a = z.get_pdb_xyz_ca(pdb_files["6NT5"])
        voxel3d_a, corner_a = z.fill_voxel_by_weight_density(
            xyz_a,
            aa_name_list_a,
            global_params["residue_weight_map"],
            grid_width,
            residue_box[grid_width],
        )
        center_scaled_a, ab_list_a, zm_list_a = one_time_conversion(
            voxel3d_a, corner_a, grid_width, cache_data, global_params
        )

        xyz_b, aa_name_list_b = z.get_pdb_xyz_ca(pdb_files["6NT6"])
        voxel3d_b, corner_b = z.fill_voxel_by_weight_density(
            xyz_b,
            aa_name_list_b,
            global_params["residue_weight_map"],
            grid_width,
            residue_box[grid_width],
        )
        center_scaled_b, ab_list_b, zm_list_b = one_time_conversion(
            voxel3d_b, corner_b, grid_width, cache_data, global_params
        )

        # Calculate similarity
        m = np.abs(zm_list_a.conj().T @ zm_list_b)
        max_similarity = np.max(m)

        # Assert that maximum similarity is significant
        # (structures should have some similarity)
        assert max_similarity > 0.1

        # Save similarity score
        with open(output_dir / "similarity_score.txt", "w") as f:
            f.write(f"Maximum similarity score: {max_similarity}\n")
            f.write(f"Similarity matrix shape: {m.shape}\n")
            f.write(f"Mean similarity: {np.mean(m)}\n")
            f.write(f"Std similarity: {np.std(m)}\n")
