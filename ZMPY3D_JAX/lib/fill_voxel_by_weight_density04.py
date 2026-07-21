# The initial bounding box calculations are straightforward to convert to JAX.
# The loop over `num_of_atom` is a challenge for direct JAX transformation, as JAX prefers vectorized operations. This loop would ideally be replaced with `jax.lax.scan` or a custom JAX primitive if the operations within the loop can be made fully JAX-compatible.
# The dictionary lookups for `aa_name` and `residue_box` would need to be handled carefully, potentially by converting them to JAX arrays and using `jax.numpy.take` or similar indexing operations.
# The array slicing and in-place addition on `voxel3d` are also areas that need careful JAX conversion, as JAX arrays are immutable. This would likely involve `jax.lax.dynamic_update_slice` or similar functional updates.

from typing import Dict, Sequence, Tuple

import chex
import jax.numpy as jnp
import numpy as np

import ZMPY3D_JAX.config as _config


def fill_voxel_by_weight_density_host(
    xyz: chex.Array,
    aa_name_list: Sequence[str],
    residue_weight_map: Dict[str, float],
    grid_width: float,
    residue_box: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fills a 3D voxel grid with density values based on atomic coordinates, amino acid types,
    and pre-calculated residue density boxes. This effectively converts a discrete atomic
    structure into a continuous density map.

    Args:
        xyz (np.ndarray): A NumPy array of shape (N, 3) with atomic coordinates.
        aa_name_list (list): A list of three-letter amino acid codes corresponding to the atoms.
        residue_weight_map (dict): A dictionary mapping three-letter amino acid codes to their weights.
        grid_width (float): The width of each voxel grid cell.
        residue_box (dict): A dictionary mapping residue names to their 3D Gaussian density boxes.

    Returns:
        tuple: A tuple containing:
            - voxel3d (np.ndarray): A 3D NumPy array representing the filled voxel grid.
            - corner_xyz (np.ndarray): A 1D NumPy array representing the corner coordinates of the voxel grid.
    """
    xyz = np.asarray(xyz, dtype=_config.FLOAT_DTYPE).reshape((-1, 3))

    if len(aa_name_list) != xyz.shape[0]:
        raise ValueError("aa_name_list must contain one residue name per coordinate")

    if xyz.shape[0] == 0:
        return np.zeros((0, 0, 0), dtype=_config.FLOAT_DTYPE), np.full(
            (3,), np.nan, dtype=_config.FLOAT_DTYPE
        )

    unknown = sorted(
        {
            name
            for name in aa_name_list
            if name not in residue_weight_map or name not in residue_box
        }
    )
    if unknown:
        raise ValueError(f"Unknown residue name(s): {', '.join(unknown)}")

    # This is intentionally a host-side kernel. Normalize each required box once,
    # rather than triggering a JAX-to-NumPy conversion for every atom placement.
    host_residue_box = {
        name: np.asarray(residue_box[name], dtype=_config.FLOAT_DTYPE)
        for name in set(aa_name_list)
    }

    min_bbox_point = np.min(xyz, axis=0)
    max_bbox_point = np.max(xyz, axis=0)
    dimension_bbox_unscaled = max_bbox_point - min_bbox_point

    max_box_edge = max([box.shape[0] for box in residue_box.values()])

    dimension_bbox_scaled = np.ceil(
        (dimension_bbox_unscaled / grid_width) + max_box_edge
    ).astype(int)
    corner_xyz = min_bbox_point - max_box_edge * grid_width / 2

    num_of_atom = xyz.shape[0]

    voxel3d = np.zeros(dimension_bbox_scaled, dtype=_config.FLOAT_DTYPE)

    for i in range(num_of_atom):
        aa_name = aa_name_list[i]
        coord = xyz[i, :]
        aa_box = host_residue_box[aa_name]
        box_edge = aa_box.shape[0]

        coord_box_corner = np.trunc(
            np.round((coord - corner_xyz) / grid_width - box_edge / 2)
        ).astype(int)

        start = coord_box_corner
        end = coord_box_corner + box_edge

        voxel3d[start[0] : end[0], start[1] : end[1], start[2] : end[2]] += aa_box

    return voxel3d, np.asarray(corner_xyz, dtype=_config.FLOAT_DTYPE)


def fill_voxel_by_weight_density04(
    xyz: chex.Array,
    aa_name_list: Sequence[str],
    residue_weight_map: Dict[str, float],
    grid_width: float,
    residue_box: Dict[str, np.ndarray],
) -> Tuple[chex.Array, chex.Array]:
    """Fill a voxel grid on the host and transfer the completed arrays to JAX."""
    voxel3d, corner_xyz = fill_voxel_by_weight_density_host(
        xyz, aa_name_list, residue_weight_map, grid_width, residue_box
    )
    return jnp.asarray(voxel3d, dtype=_config.FLOAT_DTYPE), jnp.asarray(
        corner_xyz, dtype=_config.FLOAT_DTYPE
    )
