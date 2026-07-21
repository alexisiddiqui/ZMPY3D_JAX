import numpy as np
import pytest

from ZMPY3D_JAX.lib.batched_descriptor import pad_voxel_batch


def test_pad_voxel_batch_uses_high_side_zero_padding():
    first = np.ones((2, 3, 4))
    second = np.full((3, 2, 2), 2.0)

    result = pad_voxel_batch([first, second])

    assert result.shape == (2, 3, 3, 4)
    np.testing.assert_array_equal(result[0, :2, :3, :4], first)
    np.testing.assert_array_equal(result[1, :3, :2, :2], second)
    assert np.all(result[0, 2] == 0)
    assert np.all(result[1, :, 2] == 0)
    assert np.all(result[1, :, :, 2:] == 0)


@pytest.mark.parametrize(
    "voxels, message",
    [
        ([], "non-empty"),
        ([np.ones((2, 2))], "rank-3"),
        ([np.zeros((2, 2, 2))], "zero-size array"),
    ],
)
def test_pad_voxel_batch_rejects_invalid_inputs(voxels, message):
    with pytest.raises(ValueError, match=message):
        pad_voxel_batch(voxels)
