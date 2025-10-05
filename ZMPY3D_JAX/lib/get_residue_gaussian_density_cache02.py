# The core numerical operations (summation, division, exponentiation, array multiplication) are all JAX-compatible.
# The loops can be handled by `jax.lax.scan` or by ensuring that `calculate_box_by_grid_width` is JAX-transformed and then mapping over the `doable_grid_width_list` if possible.
# The dictionary creation would remain outside the JAX-transformed functions, but the arrays within them would be JAX arrays.

from typing import Any, Dict

import numpy as np

from .calculate_box_by_grid_width import *


def get_residue_gaussian_density_cache02(
    param: Dict[str, Any],
) -> Dict[float, Dict[str, np.ndarray]]:
    """Generates a cache of 3D Gaussian density boxes for different amino acid residues
    and grid widths. This pre-calculates the density distributions used for voxelization.

    Args:
        param (dict): A dictionary containing global parameters, including 'residue_name'
                      and 'doable_grid_width_list'.

    Returns:
        dict: A dictionary where keys are grid widths and values are dictionaries mapping
              residue names to their corresponding 3D Gaussian density boxes.
    """
    residue_name = param["residue_name"]
    doable_grid_width_list = param["doable_grid_width_list"]

    aa_box_list_gw_1 = calculate_box_by_grid_width(param, 1.00)

    total_gaussian_density_gw_1 = [np.sum(box) for box in aa_box_list_gw_1]

    aa_box_map_ref = dict(zip(residue_name, aa_box_list_gw_1))

    grid_widths = [1] + doable_grid_width_list

    aa_box_map_list = [aa_box_map_ref]

    for gw in grid_widths[1:]:
        temp_aa_box_list = calculate_box_by_grid_width(param, gw)

        temp_total_gaussian_density = [np.sum(box) for box in temp_aa_box_list]

        density_scalar = [
            td_gw1 / (td * gw**3)
            for td_gw1, td in zip(total_gaussian_density_gw_1, temp_total_gaussian_density)
        ]

        adjusted_temp_aa_box_list = [
            box * scalar for box, scalar in zip(temp_aa_box_list, density_scalar)
        ]

        aa_box_map_list.append(dict(zip(residue_name, adjusted_temp_aa_box_list)))

    residue_box = dict(zip(grid_widths, aa_box_map_list))

    return residue_box
