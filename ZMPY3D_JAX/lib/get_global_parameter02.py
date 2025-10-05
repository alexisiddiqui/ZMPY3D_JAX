# This function primarily sets up static parameters.
# The numerical calculation for `three_over_2pi_32` can be done using `jax.numpy.pi` and `jax.numpy.power`.
# The dictionaries themselves would remain Python dictionaries, but their values (weights, radii) could be converted to JAX arrays if they are to be used in JAX-transformed functions.


import math
from typing import Any, Dict

from .get_residue_radius_map01 import *
from .get_residue_weight_map01 import *


def get_global_parameter02() -> Dict[str, Any]:
    """Initializes and returns a dictionary containing various global parameters
    used throughout the ZMPY3D codebase, such as grid widths, radius multipliers,
    density calculation constants, and mappings for residue weights and radii.

    Returns:
        dict: A dictionary of global parameters.
    """
    param = {}

    param["doable_grid_width_list"] = [0.25, 0.5, 2, 4, 8, 16]
    param["default_radius_multiplier"] = 1.8

    sd_cutoff = 3.0
    three_over_2pi_32 = (3 / (2 * math.pi)) ** (3 / 2)
    density_multiplier = 100

    residue_name = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "MSE",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "A",
        "T",
        "C",
        "G",
        "U",
        "I",
        "DA",
        "DT",
        "DG",
        "DC",
        "DU",
        "DI",
    ]

    residue_weight_map = get_residue_weight_map01()
    residue_radius_map = get_residue_radius_map01()

    param["sd_cutoff"] = sd_cutoff
    param["three_over_2pi_32"] = three_over_2pi_32
    param["density_multiplier"] = density_multiplier
    param["residue_name"] = residue_name
    param["residue_weight_map"] = residue_weight_map
    param["residue_radius_map"] = residue_radius_map

    return param
