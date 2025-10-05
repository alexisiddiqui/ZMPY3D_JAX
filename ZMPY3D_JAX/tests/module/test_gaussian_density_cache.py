from unittest.mock import patch

import numpy as np
import pytest

from ZMPY3D_JAX.lib.get_residue_gaussian_density_cache02 import get_residue_gaussian_density_cache02


@pytest.fixture
def basic_param():
    """Basic parameter dictionary for testing."""
    return {
        "residue_name": ["ALA", "GLY", "VAL"],
        "doable_grid_width_list": [1.5, 2.0, 2.5],
    }


@pytest.fixture
def mock_boxes():
    """Mock 3D boxes for different residues."""
    return [
        np.ones((5, 5, 5)) * 0.8,  # ALA
        np.ones((4, 4, 4)) * 0.6,  # GLY
        np.ones((6, 6, 6)) * 0.9,  # VAL
    ]


def test_get_residue_gaussian_density_cache_structure(basic_param, mock_boxes):
    """Test that the function returns the correct structure."""
    with patch(
        "ZMPY3D_JAX.lib.get_residue_gaussian_density_cache02.calculate_box_by_grid_width"
    ) as mock_calc:
        mock_calc.return_value = mock_boxes

        result = get_residue_gaussian_density_cache02(basic_param)

        # Check that result is a dictionary
        assert isinstance(result, dict)

        # Check that grid widths are keys (1.0 + doable_grid_width_list)
        expected_widths = [1] + basic_param["doable_grid_width_list"]
        assert set(result.keys()) == set(expected_widths)

        # Check that each grid width maps to a residue dictionary
        for gw in expected_widths:
            assert isinstance(result[gw], dict)
            assert set(result[gw].keys()) == set(basic_param["residue_name"])


def test_get_residue_gaussian_density_cache_arrays(basic_param, mock_boxes):
    """Test that all values are numpy arrays."""
    with patch(
        "ZMPY3D_JAX.lib.get_residue_gaussian_density_cache02.calculate_box_by_grid_width"
    ) as mock_calc:
        mock_calc.return_value = mock_boxes

        result = get_residue_gaussian_density_cache02(basic_param)

        for gw_dict in result.values():
            for residue_box in gw_dict.values():
                assert isinstance(residue_box, np.ndarray)


def test_density_scaling_conservation(basic_param):
    """Test that density is properly scaled across different grid widths."""
    # Create mock boxes with known sums
    box_gw1 = [
        np.ones((5, 5, 5)) * 1.0,  # sum = 125
        np.ones((4, 4, 4)) * 0.5,  # sum = 32
    ]
    box_gw2 = [
        np.ones((3, 3, 3)) * 2.0,  # sum = 54
        np.ones((2, 2, 2)) * 1.0,  # sum = 8
    ]

    param = {
        "residue_name": ["ALA", "GLY"],
        "doable_grid_width_list": [2.0],
    }

    with patch(
        "ZMPY3D_JAX.lib.get_residue_gaussian_density_cache02.calculate_box_by_grid_width"
    ) as mock_calc:
        mock_calc.side_effect = [box_gw1, box_gw2]

        result = get_residue_gaussian_density_cache02(param)

        # Verify scaling: td_gw1 / (td * gw^3)
        # For ALA at gw=2: 125 / (54 * 8) = 125/432
        expected_scalar_ala = 125 / (54 * (2**3))
        actual_sum_ala = np.sum(result[2.0]["ALA"])
        expected_sum_ala = 54 * expected_scalar_ala

        np.testing.assert_allclose(actual_sum_ala, expected_sum_ala, rtol=1e-10)


def test_reference_grid_width_unchanged(basic_param, mock_boxes):
    """Test that the reference grid width (1.0) boxes are not scaled."""
    with patch(
        "ZMPY3D_JAX.lib.get_residue_gaussian_density_cache02.calculate_box_by_grid_width"
    ) as mock_calc:
        mock_calc.return_value = mock_boxes

        result = get_residue_gaussian_density_cache02(basic_param)

        # The boxes at grid width 1.0 should be identical to the original
        for i, residue in enumerate(basic_param["residue_name"]):
            np.testing.assert_array_equal(result[1][residue], mock_boxes[i])


def test_empty_doable_grid_width_list():
    """Test with empty doable_grid_width_list."""
    param = {
        "residue_name": ["ALA"],
        "doable_grid_width_list": [],
    }
    mock_box = [np.ones((3, 3, 3))]

    with patch(
        "ZMPY3D_JAX.lib.get_residue_gaussian_density_cache02.calculate_box_by_grid_width"
    ) as mock_calc:
        mock_calc.return_value = mock_box

        result = get_residue_gaussian_density_cache02(param)

        # Should only have grid width 1
        assert list(result.keys()) == [1]
        assert "ALA" in result[1]


def test_calculate_box_called_correctly(basic_param):
    """Test that calculate_box_by_grid_width is called with correct arguments."""
    with patch(
        "ZMPY3D_JAX.lib.get_residue_gaussian_density_cache02.calculate_box_by_grid_width"
    ) as mock_calc:
        mock_calc.return_value = [np.ones((3, 3, 3))] * 3

        get_residue_gaussian_density_cache02(basic_param)

        # Should be called for 1.0 and each grid width in doable_grid_width_list
        expected_calls = len(basic_param["doable_grid_width_list"]) + 1
        assert mock_calc.call_count == expected_calls

        # Verify it's called with correct grid widths
        call_args = [call[0][1] for call in mock_calc.call_args_list]
        expected_gw = [1.0] + basic_param["doable_grid_width_list"]
        assert call_args == expected_gw
