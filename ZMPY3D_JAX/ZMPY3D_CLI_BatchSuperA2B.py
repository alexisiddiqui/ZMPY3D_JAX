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
from typing import List, Sequence

import numpy as np

from .lib.superposition import (
    calculate_pdb_superposition,
    prepare_superposition_runtime,
)


def ZMPY3D_CLI_BatchSuperA2B(
    PDBFileNameA: Sequence[str], PDBFileNameB: Sequence[str]
) -> List[np.ndarray]:
    """Return one NumPy A-to-B transform for each legacy-PDB CA-trace pair.

    Pairs remain sequential, matching the original batch behavior, while all pairs
    share one prepared cache runtime and the same device-native matching core.
    """
    if len(PDBFileNameA) != len(PDBFileNameB):
        raise ValueError("PDBFileNameA and PDBFileNameB must have the same length")

    runtime = prepare_superposition_runtime()
    matrices = []
    for file_name_a, file_name_b in zip(PDBFileNameA, PDBFileNameB):
        match = calculate_pdb_superposition(file_name_a, file_name_b, runtime)
        matrix = np.asarray(match.matrix)
        if not bool(np.asarray(match.is_valid)):
            raise ValueError("no valid rotation candidates were generated")
        if not np.all(np.isfinite(matrix)):
            raise np.linalg.LinAlgError("superposition transform is not finite")
        matrices.append(matrix)
    return matrices


def main() -> None:
    """Read whitespace-separated PDB pairs and print their transformation matrices."""
    if len(sys.argv) != 2:
        print("Usage: ZMPY3D_CLI_BatchSuperA2B PDBFileList.txt")
        print(
            "       This function takes paired PDB paths and generates "
            "transformation matrices."
        )
        print("Error: You must provide exactly one input file.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Process a text file containing paired .pdb or .txt paths."
    )
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()

    if not args.input_file.endswith(".txt"):
        parser.error("File must end with .txt")
    if not os.path.isfile(args.input_file):
        parser.error("File does not exist")

    file_list_a = []
    file_list_b = []
    with open(args.input_file, "r", encoding="utf-8") as file:
        for line in file:
            files = line.strip().split()
            if len(files) != 2:
                parser.error(
                    f"Each line must contain exactly two file paths, got {len(files)}"
                )
            file_a, file_b = files
            for path in (file_a, file_b):
                if not (path.endswith(".pdb") or path.endswith(".txt")):
                    parser.error(f"File {path} must end with .pdb or .txt")
                if not os.path.isfile(path):
                    parser.error(f"File {path} does not exist")
            file_list_a.append(file_a)
            file_list_b.append(file_b)

    for matrix in ZMPY3D_CLI_BatchSuperA2B(file_list_a, file_list_b):
        print(matrix)


if __name__ == "__main__":
    main()
