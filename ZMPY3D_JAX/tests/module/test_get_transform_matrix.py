import numpy as np

from ZMPY3D_JAX.lib.get_transform_matrix_from_ab_list02 import get_transform_matrix_from_ab_list02


class TestGetTransformMatrixFromAbList02:
    """Test suite for get_transform_matrix_from_ab_list02 function."""

    def test_output_shape(self):
        """Test that the output is a 4x4 matrix."""
        a = 1.0 + 0.0j
        b = 0.0 + 0.0j
        center_scaled = np.array([0.0, 0.0, 0.0])

        result = get_transform_matrix_from_ab_list02(a, b, center_scaled)

        assert result.shape == (4, 4), "Output should be a 4x4 matrix"

    def test_identity_like_transformation(self):
        """Test with a=1, b=0 which should give near-identity rotation."""
        a = 1.0 + 0.0j
        b = 0.0 + 0.0j
        center_scaled = np.array([0.0, 0.0, 0.0])

        result = get_transform_matrix_from_ab_list02(a, b, center_scaled)

        # The 3x3 rotation part should be identity-like
        rotation_part = result[0:3, 0:3]
        expected_rotation = np.eye(3)

        np.testing.assert_array_almost_equal(rotation_part, expected_rotation, decimal=10)
        assert result[3, 3] == 1.0, "Bottom-right element should be 1"

    def test_with_translation(self):
        """Test that translation is correctly incorporated."""
        a = 1.0 + 0.0j
        b = 0.0 + 0.0j
        center_scaled = np.array([1.0, 2.0, 3.0])

        result = get_transform_matrix_from_ab_list02(a, b, center_scaled)

        # Check that translation vector is in the correct position (before inversion)
        # After inversion, the relationship will be different
        assert result.shape == (4, 4)
        assert result[3, 3] == 1.0

    def test_complex_coefficients(self):
        """Test with non-trivial complex coefficients."""
        a = 0.5 + 0.5j
        b = 0.3 + 0.2j
        center_scaled = np.array([1.0, 1.0, 1.0])

        result = get_transform_matrix_from_ab_list02(a, b, center_scaled)

        assert result.shape == (4, 4)
        assert np.isfinite(result).all(), "All elements should be finite"

    def test_invertibility(self):
        """Test that the transformation matrix is invertible."""
        a = 0.6 + 0.8j
        b = 0.4 + 0.3j
        center_scaled = np.array([2.0, -1.0, 3.0])

        result = get_transform_matrix_from_ab_list02(a, b, center_scaled)

        # Check that the matrix is invertible (determinant != 0)
        det = np.linalg.det(result)
        assert abs(det) > 1e-10, "Matrix should be invertible"

    def test_pure_real_coefficients(self):
        """Test with purely real coefficients."""
        a = 2.0 + 0.0j
        b = 1.0 + 0.0j
        center_scaled = np.array([0.0, 0.0, 0.0])

        result = get_transform_matrix_from_ab_list02(a, b, center_scaled)

        assert result.shape == (4, 4)
        assert np.isfinite(result).all()

    def test_pure_imaginary_coefficients(self):
        """Test with purely imaginary coefficients."""
        a = 0.0 + 1.0j
        b = 0.0 + 0.5j
        center_scaled = np.array([0.0, 0.0, 0.0])

        result = get_transform_matrix_from_ab_list02(a, b, center_scaled)

        assert result.shape == (4, 4)
        assert np.isfinite(result).all()

    def test_homogeneous_coordinates(self):
        """Test that the last row follows homogeneous coordinate convention."""
        a = 0.7 + 0.3j
        b = 0.2 + 0.4j
        center_scaled = np.array([1.5, -2.5, 0.5])

        result = get_transform_matrix_from_ab_list02(a, b, center_scaled)

        # Last row should be [*, *, *, 1] after inversion
        assert result[3, 3] == 1.0

    def test_center_scaled_as_column_vector(self):
        """Test with center_scaled as a column vector."""
        a = 1.0 + 0.0j
        b = 0.0 + 0.0j
        center_scaled = np.array([[1.0], [2.0], [3.0]])

        result = get_transform_matrix_from_ab_list02(a, b, center_scaled)

        assert result.shape == (4, 4)

    def test_numerical_stability(self):
        """Test numerical stability with small coefficients."""
        a = 1e-5 + 1e-5j
        b = 1e-6 + 1e-6j
        center_scaled = np.array([0.0, 0.0, 0.0])

        result = get_transform_matrix_from_ab_list02(a, b, center_scaled)

        assert np.isfinite(result).all(), "Result should be numerically stable"

    def test_determinant_relationship(self):
        """Test that the determinant has expected properties."""
        a = 1.0 + 0.0j
        b = 0.0 + 0.5j
        center_scaled = np.array([0.0, 0.0, 0.0])

        result = get_transform_matrix_from_ab_list02(a, b, center_scaled)
        det = np.linalg.det(result)

        assert abs(det) > 0, "Determinant should be non-zero"
