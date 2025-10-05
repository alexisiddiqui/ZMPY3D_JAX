# This function defines a static dictionary and performs a simple scalar multiplication (`math.sqrt(5.0 / 3.0) * value`).
# In a JAX context, similar to `get_residue_weight_map01`, this dictionary would likely be converted into a static JAX array or a lookup table if its values are to be used within JAX-transformed functions.
# The `math.sqrt` operation is a standard numerical operation.

import math
from typing import Dict


def get_residue_radius_map01() -> Dict[str, float]:
    """Returns a dictionary mapping three-letter amino acid codes (and some nucleotide codes)
    to their corresponding residue radii, scaled by a constant factor sqrt(5/3).

    Returns:
        dict: A dictionary with amino acid/nucleotide codes as keys and their scaled radii as values.
    """
    coef = math.sqrt(5.0 / 3.0)

    residue_radii = {
        "ALA": 1.963913 * coef,
        "ARG": 3.374007 * coef,
        "ASN": 2.695111 * coef,
        "ASP": 2.525241 * coef,
        "CYS": 2.413249 * coef,
        "GLN": 3.088783 * coef,
        "GLU": 2.883527 * coef,
        "GLY": 1.841949 * coef,
        "HIS": 2.652737 * coef,
        "ILE": 2.575828 * coef,
        "LEU": 2.736953 * coef,
        "LYS": 3.177825 * coef,
        "MET": 2.959014 * coef,
        "MSE": 2.959014 * coef,
        "PHE": 2.979213 * coef,
        "PRO": 2.266054 * coef,
        "SER": 2.184637 * coef,
        "THR": 2.366486 * coef,
        "TRP": 3.248871 * coef,
        "TYR": 3.217711 * coef,
        "VAL": 2.351359 * coef,
        "A": 4.333750 * coef,
        "T": 3.700942 * coef,
        "G": 4.443546 * coef,
        "C": 3.954067 * coef,
        "U": 3.964129 * coef,
        "I": 4.0 * coef,
        "DA": 4.333750 * coef,
        "DT": 3.700942 * coef,
        "DG": 4.443546 * coef,
        "DC": 3.954067 * coef,
        "DU": 3.964129 * coef,
        "DI": 4.0 * coef,
    }

    return residue_radii
