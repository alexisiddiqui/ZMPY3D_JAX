# The matrix multiplication and rounding are prime candidates for JAX transformation (e.g., `jit`, `grad`).
# The string parsing and formatting parts would remain standard Python/NumPy operations.

import numpy as np

from .read_file_as_string_list import read_file_as_string_list
from .write_string_list_as_file import write_string_list_as_file


def set_pdb_xyz_rot_m_01(input_file_name: str, rot_m: np.ndarray, output_file_name: str) -> None:
    """Applies a 3D rotation matrix to the atomic coordinates (X, Y, Z) within a PDB file
    and writes the transformed coordinates to a new PDB file.

    Args:
        input_file_name (str): The path to the input PDB file.
        rot_m (np.ndarray): A 4x4 rotation matrix to be applied to the coordinates.
        output_file_name (str): The path to the output PDB file where transformed coordinates will be written.
    """

    def sprintf2(format_spec, numbers):
        return [format_spec % num for num in numbers]

    contents = read_file_as_string_list(input_file_name)

    x = [float(s[30:38]) for s in contents if s.startswith("ATOM")]
    y = [float(s[38:46]) for s in contents if s.startswith("ATOM")]
    z = [float(s[46:54]) for s in contents if s.startswith("ATOM")]

    target_xyz = np.array([x, y, z]).T

    temp_xyz = rot_m @ np.hstack((target_xyz, np.ones((target_xyz.shape[0], 1)))).T
    temp_xyz = np.round(temp_xyz, decimals=3)

    x_formatted = sprintf2("%8.3f", temp_xyz[0, :])
    y_formatted = sprintf2("%8.3f", temp_xyz[1, :])
    z_formatted = sprintf2("%8.3f", temp_xyz[2, :])

    atom_count = 0
    for i, s in enumerate(contents):
        if s.startswith("ATOM"):
            contents[i] = (
                s[:30]
                + x_formatted[atom_count]
                + y_formatted[atom_count]
                + z_formatted[atom_count]
                + s[54:]
            )
            atom_count += 1

    write_string_list_as_file(contents, output_file_name)
