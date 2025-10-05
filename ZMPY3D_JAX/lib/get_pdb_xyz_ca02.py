# The primary numerical part is the creation of the `xyz_matrix` NumPy array.
# If the parsing and filtering steps can be made JAX-compatible (e.g., by operating on string arrays or pre-parsed numerical data), then the `xyz_matrix` could directly be a JAX array.
# The NaN check could use `jax.numpy.isnan`.
# However, the string parsing itself is not directly JAX-transformable and would likely remain a Python/NumPy preprocessing step.

import math
from typing import List, Tuple

import numpy as np


def get_pdb_xyz_ca02(file_name: str) -> Tuple[np.ndarray, List[str]]:
    """Parses a PDB file to extract the XYZ coordinates and amino acid names
    specifically for C-alpha (CA) atoms. It also checks for NaN values in coordinates.

    Args:
        file_name (str): The path to the input PDB file.

    Returns:
        tuple: A tuple containing:
            - xyz_matrix (np.ndarray): A NumPy array of shape (N, 3) with C-alpha atom coordinates.
            - aa_names (list): A list of three-letter amino acid codes corresponding to the C-alpha atoms.

    Raises:
        ValueError: If any NaN values are found in the extracted coordinates.
    """
    with open(file_name, "r") as file:
        lines = file.readlines()

    atom_lines = [line for line in lines if line.startswith("ATOM")]

    ca_lines = [line for line in atom_lines if line[13:15] == "CA"]

    xyz = []
    aa_names = []
    for line in ca_lines:
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        aa_name = line[17:20].strip()

        xyz.append((x, y, z))
        aa_names.append(aa_name)

        if any(map(math.isnan, [x, y, z])):
            raise ValueError("has nan in XYZ")

    xyz_matrix = np.array(xyz)

    return xyz_matrix, aa_names
