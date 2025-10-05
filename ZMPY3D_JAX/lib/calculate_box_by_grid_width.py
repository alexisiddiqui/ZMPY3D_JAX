# The loop over `residue_name` should be vectorized using `jax.vmap` if possible, or handled with `jax.lax.scan`.
# `np.mgrid` can be replaced with `jax.numpy.mgrid`.
# All numerical operations (`np.sqrt`, `**`, division, multiplication, `np.exp`, comparisons) have direct `jax.numpy` equivalents.
# Boolean indexing and conditional assignment can be handled with `jax.numpy.where` or `jax.lax.select` to maintain immutability.


from typing import Any, Dict, List

import chex
import jax.numpy as jnp
import numpy as np

import ZMPY3D_JAX.config as _config


def calculate_box_by_grid_width(param: Dict[str, Any], grid_width: float) -> List[chex.Array]:
    """Generates 3D Gaussian density boxes for each amino acid residue at a specified grid width.
    These boxes represent the spatial density distribution of each residue.

    Args:
        param (dict): A dictionary containing global parameters, including residue maps and density constants.
        grid_width (float): The width of each voxel grid cell.

    Returns:
        list: A list of 3D NumPy arrays, where each array is a Gaussian density box for a residue.
    """
    residue_name = param["residue_name"]
    residue_weight_map = param["residue_weight_map"]
    residue_radius_map = param["residue_radius_map"]

    sd_cutoff = param["sd_cutoff"]
    three_over_2pi_cube_root = param["three_over_2pi_32"]
    density_multiplier = param["density_multiplier"]

    aa_box_list = []

    for aa in residue_name:
        weight = residue_weight_map[aa]
        radius = residue_radius_map[aa]

        sigma = np.sqrt(radius**2 / 5.0)
        box_edge = int(np.ceil(2 * sd_cutoff * sigma / grid_width))

        if box_edge % 2 == 0:
            box_edge += 1

        center = box_edge // 2
        sqr_radius = center**2

        x, y, z = np.mgrid[0:box_edge, 0:box_edge, 0:box_edge] - center

        is_within_radius = x**2 + y**2 + z**2 <= sqr_radius

        x_sx = (x**2 + y**2 + z**2) * grid_width**2 / (radius**2 / 5.0)
        gaus_val = (
            weight
            * three_over_2pi_cube_root
            / (radius**2 / 5.0 * sigma)
            * np.exp(-0.5 * x_sx)
            * density_multiplier
        )

        residue_unit_box = np.zeros((box_edge, box_edge, box_edge))
        residue_unit_box[is_within_radius] = gaus_val[is_within_radius]

        aa_box_list.append(jnp.asarray(residue_unit_box, dtype=_config.FLOAT_DTYPE))

    return aa_box_list
