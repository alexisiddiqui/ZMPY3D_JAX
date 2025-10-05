"""
Tests for fill_voxel_by_weight_density function.
"""

import sys
from pathlib import Path

import chex
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import ZMPY3D_JAX as z


class TestFillVoxelByWeightDensity:
    """Test suite for fill_voxel_by_weight_density function."""

    @pytest.fixture
    def param(self):
        """Get global parameters."""
        return z.get_global_parameter()

    @pytest.fixture
    def residue_box_cache(self, param):
        """Get residue gaussian density cache for a specific grid width."""
        return z.get_residue_gaussian_density_cache(param)

    @pytest.fixture
    def sample_coords_single_atom(self):
        """Sample coordinates for a single atom."""
        return np.array([[10.0, 10.0, 10.0]])

    @pytest.fixture
    def sample_aa_single_atom(self):
        """Sample amino acid name for a single atom."""
        return ["ALA"]

    @pytest.fixture
    def sample_coords_multiple_atoms(self):
        """Sample coordinates for multiple atoms."""
        return np.array(
            [
                [10.0, 10.0, 10.0],
                [11.0, 10.0, 10.0],
                [10.0, 11.0, 10.0],
                [10.0, 10.0, 11.0],
            ]
        )

    @pytest.fixture
    def sample_aa_multiple_atoms(self):
        """Sample amino acid names for multiple atoms."""
        return ["ALA", "GLY", "VAL", "LEU"]

    def test_single_atom_filling(
        self, param, residue_box_cache, sample_coords_single_atom, sample_aa_single_atom
    ):
        grid_width = 1.0
        voxel3d, corner_xyz = z.fill_voxel_by_weight_density(
            sample_coords_single_atom,
            sample_aa_single_atom,
            param["residue_weight_map"],
            grid_width,
            residue_box_cache[grid_width],
        )

        assert isinstance(voxel3d, chex.Array)
        assert isinstance(corner_xyz, chex.Array)
        assert voxel3d.ndim == 3
        assert corner_xyz.shape == (3,)

        # Check that some density is present
        assert np.sum(voxel3d) > 0

        # Check that the center of the density is roughly where the atom was
        # This is an approximation due to voxelization and Gaussian spread
        density_center = np.array(np.where(voxel3d == np.max(voxel3d))) * grid_width + corner_xyz
        assert np.allclose(density_center.mean(axis=1), sample_coords_single_atom[0], atol=2.0)

    def test_multiple_atoms_filling(
        self, param, residue_box_cache, sample_coords_multiple_atoms, sample_aa_multiple_atoms
    ):
        """Test filling with multiple atoms of different types."""
        grid_width = 1.0
        voxel3d, corner_xyz = z.fill_voxel_by_weight_density(
            sample_coords_multiple_atoms,
            sample_aa_multiple_atoms,
            param["residue_weight_map"],
            grid_width,
            residue_box_cache[grid_width],
        )

        assert np.sum(voxel3d) > 0
        assert voxel3d.ndim == 3
        assert corner_xyz.shape == (3,)

    def test_empty_input(self, param, residue_box_cache):
        """Test with empty atomic coordinates."""
        empty_xyz = np.array([]).reshape(0, 3)
        empty_aa_names = []
        grid_width = 1.0

        voxel3d, corner_xyz = z.fill_voxel_by_weight_density(
            empty_xyz,
            empty_aa_names,
            param["residue_weight_map"],
            grid_width,
            residue_box_cache[grid_width],
        )

        # Should return an empty-like voxel and corner_xyz (or default values)
        assert voxel3d.shape == (0, 0, 0) or np.all(voxel3d == 0)
        assert corner_xyz.shape == (3,)
        assert np.all(np.isnan(corner_xyz)) or np.all(corner_xyz == 0)

    def test_different_grid_width(
        self, param, residue_box_cache, sample_coords_single_atom, sample_aa_single_atom
    ):
        """Test with a different grid width."""
        grid_width = 0.5
        voxel3d, corner_xyz = z.fill_voxel_by_weight_density(
            sample_coords_single_atom,
            sample_aa_single_atom,
            param["residue_weight_map"],
            grid_width,
            residue_box_cache[grid_width],
        )

        assert np.sum(voxel3d) > 0
        assert voxel3d.ndim == 3
        assert corner_xyz.shape == (3,)

    def test_unknown_amino_acid(self, param, residue_box_cache, sample_coords_single_atom):
        """Test with an unknown amino acid, which should default to ASP."""
        unknown_aa = ["XXX"]
        grid_width = 1.0

        voxel3d, corner_xyz = z.fill_voxel_by_weight_density(
            sample_coords_single_atom,
            unknown_aa,
            param["residue_weight_map"],
            grid_width,
            residue_box_cache[grid_width],
        )

        assert np.sum(voxel3d) > 0
        assert voxel3d.ndim == 3
        assert corner_xyz.shape == (3,)

        # To be more precise, we could compare the sum of densities to that of an ASP residue
        # This would require calculating the expected sum for ASP, which is complex.
        # For now, just checking that density is generated is sufficient.

    def test_deterministic_output(
        self, param, residue_box_cache, sample_coords_multiple_atoms, sample_aa_multiple_atoms
    ):
        """Test that the function produces deterministic output."""
        grid_width = 1.0

        voxel3d_1, corner_xyz_1 = z.fill_voxel_by_weight_density(
            sample_coords_multiple_atoms,
            sample_aa_multiple_atoms,
            param["residue_weight_map"],
            grid_width,
            residue_box_cache[grid_width],
        )

        voxel3d_2, corner_xyz_2 = z.fill_voxel_by_weight_density(
            sample_coords_multiple_atoms,
            sample_aa_multiple_atoms,
            param["residue_weight_map"],
            grid_width,
            residue_box_cache[grid_width],
        )

        np.testing.assert_array_equal(voxel3d_1, voxel3d_2)
        np.testing.assert_array_equal(corner_xyz_1, corner_xyz_2)

    def test_voxel_dimensions_and_corner(
        self, param, residue_box_cache, sample_coords_single_atom, sample_aa_single_atom
    ):
        """Test that voxel dimensions and corner_xyz are reasonable."""
        grid_width = 1.0
        voxel3d, corner_xyz = z.fill_voxel_by_weight_density(
            sample_coords_single_atom,
            sample_aa_single_atom,
            param["residue_weight_map"],
            grid_width,
            residue_box_cache[grid_width],
        )

        # Voxel dimensions should be large enough to contain the residue box
        ala_box_shape = residue_box_cache[grid_width]["ALA"].shape
        assert voxel3d.shape[0] >= ala_box_shape[0]
        assert voxel3d.shape[1] >= ala_box_shape[1]
        assert voxel3d.shape[2] >= ala_box_shape[2]

        # Corner_xyz should be less than or equal to the min coordinate
        assert np.all(corner_xyz <= sample_coords_single_atom.min(axis=0))

    def test_density_sum_proportional_to_weight(self, param, residue_box_cache):
        """Test that total density is proportional to residue weight."""
        grid_width = 1.0
        ala_coords = np.array([[0.0, 0.0, 0.0]])
        gly_coords = np.array([[0.0, 0.0, 0.0]])

        voxel_ala, _ = z.fill_voxel_by_weight_density(
            ala_coords,
            ["ALA"],
            param["residue_weight_map"],
            grid_width,
            residue_box_cache[grid_width],
        )
        voxel_gly, _ = z.fill_voxel_by_weight_density(
            gly_coords,
            ["GLY"],
            param["residue_weight_map"],
            grid_width,
            residue_box_cache[grid_width],
        )

        sum_ala = np.sum(voxel_ala)
        sum_gly = np.sum(voxel_gly)

        # Glycine is smaller than Alanine, so its total density should be less
        # This is a general expectation, not a strict equality due to Gaussian tails
        assert sum_ala > sum_gly
        assert sum_ala > 0
        assert sum_gly > 0
