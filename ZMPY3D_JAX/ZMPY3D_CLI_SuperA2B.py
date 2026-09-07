# MIT License
#
# Copyright (c) 2024 Jhih-Siang Lai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import os
import sys

import numpy as np

from .lib.superposition import (
    calculate_pdb_superposition,
    prepare_superposition_runtime,
)


def ZMPY3D_CLI_SuperA2B(PDBFileNameA: str, PDBFileNameB: str) -> np.ndarray:
    """Return the 4x4 matrix that transforms legacy-PDB CA trace A onto B.

    Candidate generation, moment rotation, feature matching, and the final linear
    solve remain in JAX. The completed matrix crosses to NumPy at this public API
    boundary for compatibility with the original interface.
    """
    runtime = prepare_superposition_runtime()
    match = calculate_pdb_superposition(PDBFileNameA, PDBFileNameB, runtime)
    matrix = np.asarray(match.matrix)
    if not bool(np.asarray(match.is_valid)):
        raise ValueError("no valid rotation candidates were generated")
    if not np.all(np.isfinite(matrix)):
        raise np.linalg.LinAlgError("superposition transform is not finite")
    return matrix


def main() -> None:
    """Parse two legacy PDB paths and print the A-to-B transformation matrix."""
    if len(sys.argv) != 3:
        print("Usage: ZMPY3D_CLI_SuperA2B PDB_A PDB_B")
        print(
            "    This function generates a transformation matrix to superimpose "
            "structure A onto B, i.e., the matrix is for A’s coordinates."
        )
        print("Error: You must provide exactly two input files.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Process two .pdb or .txt files.")
    parser.add_argument(
        "input_file1", type=str, help="The first input file in old PDB text format"
    )
    parser.add_argument(
        "input_file2", type=str, help="The second input file in old PDB text format"
    )
    args = parser.parse_args()

    for input_file in (args.input_file1, args.input_file2):
        if not (input_file.endswith(".pdb") or input_file.endswith(".txt")):
            parser.error("File must end with .pdb or .txt")
        if not os.path.isfile(input_file):
            parser.error("File does not exist")

    target_rot_m = ZMPY3D_CLI_SuperA2B(args.input_file1, args.input_file2)
    print("the matrix is for A’s coordinates.")
    print(target_rot_m)


if __name__ == "__main__":
    main()
