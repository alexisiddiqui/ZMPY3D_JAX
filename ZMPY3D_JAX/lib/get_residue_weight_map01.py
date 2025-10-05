# This function defines a static dictionary.
# It does not involve any numerical computations.
# In a JAX context, this dictionary would likely be converted into a static JAX array or a lookup table if its values are to be used within JAX-transformed functions.


from typing import Dict


def get_residue_weight_map01() -> Dict[str, float]:
    """Returns a dictionary mapping three-letter amino acid codes (and some nucleotide codes)
    to their corresponding residue weights.

    Returns:
        dict: A dictionary with amino acid/nucleotide codes as keys and their weights as values.
    """
    residue_weight_map = {
        "ALA": 82.03854,
        "ARG": 160.09176,
        "ASN": 148.07768,
        "ASP": 126.04834,
        "CYS": 114.10454,
        "GLN": 160.08868,
        "GLU": 138.05934,
        "GLY": 70.02754,
        "HIS": 146.08502,
        "ILE": 214.15954,
        "LEU": 190.13754,
        "LYS": 132.07828,
        "MET": 138.12654,
        "MSE": 138.12654,
        "PHE": 154.10454,
        "PRO": 106.06054,
        "SER": 98.03794,
        "THR": 146.08194,
        "TRP": 192.13328,
        "TYR": 170.10394,
        "VAL": 178.12654,
        "A": 409.12186,
        "T": 379.11264,
        "G": 425.12126,
        "C": 385.09678,
        "U": 387.08944,
        "I": 400.0,
        "DA": 409.12186,
        "DT": 379.11264,
        "DG": 425.12126,
        "DC": 385.09678,
        "DU": 387.08944,
        "DI": 400.0,
    }

    return residue_weight_map
