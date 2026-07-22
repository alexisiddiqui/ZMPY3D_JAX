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
from concurrent.futures import Future, ThreadPoolExecutor
import os
import pickle
from functools import partial
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import ZMPY3D_JAX as z
from ZMPY3D_JAX.lib.batched_descriptor import (
    calculate_descriptor_batch_from_voxels,
    pad_voxel_batch,
)
from ZMPY3D_JAX.lib.fill_voxel_by_weight_density04 import (
    fill_voxel_by_weight_density_host,
)


class _BatchZMRuntime(NamedTuple):
    param: dict[str, Any]
    residue_box: dict[float, Any]
    rotation_cache: z.ZMRotationCache
    bbox_to_zm_cache: z.BBoxToZMCache
    x64_bbox_to_zm_cache: z.BBoxToZMCache | None
    descriptor_cache: z.DescriptorAssemblyCache


def _prepare_batch_runtime(grid_width: float, max_order: int) -> _BatchZMRuntime:
    param = z.get_global_parameter()
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_data")
    with open(os.path.join(cache_dir, "BinomialCache.pkl"), "rb") as file:
        binomial_cache = pickle.load(file)["BinomialCache"]
    with open(
        os.path.join(cache_dir, f"LogG_CLMCache_MaxOrder{max_order:02d}.pkl"), "rb"
    ) as file:
        cache = pickle.load(file)

    rotation_index = cache["RotationIndex"]
    rotation_cache = z.prepare_zm_rotation_cache(
        binomial_cache,
        max_order,
        cache["CLMCache"],
        np.squeeze(rotation_index["s_id"][0, 0]) - 1,
        np.squeeze(rotation_index["n"][0, 0]),
        np.squeeze(rotation_index["l"][0, 0]),
        np.squeeze(rotation_index["m"][0, 0]),
        np.squeeze(rotation_index["mu"][0, 0]),
        np.squeeze(rotation_index["k"][0, 0]),
        np.squeeze(rotation_index["IsNLM_Value"][0, 0]) - 1,
    )
    bbox_to_zm_cache = z.prepare_bbox_to_zm_cache(
        max_order,
        cache["GCache_complex"],
        cache["GCache_pqr_linear"],
        cache["GCache_complex_index"],
        cache["CLMCache3D"],
    )
    x64_bbox_to_zm_cache = None
    if z.FLOAT_DTYPE == jnp.float32 and max_order >= 20:
        jax.config.update("jax_enable_x64", True)
        x64_bbox_to_zm_cache = z.BBoxToZMCache(
            max_order=max_order,
            g_coefficients=jnp.asarray(
                cache["GCache_complex"], dtype=jnp.complex128
            ).reshape(-1),
            pqr_indices=jnp.asarray(
                cache["GCache_pqr_linear"], dtype=jnp.int32
            ).reshape(-1)
            - 1,
            output_indices=jnp.asarray(
                cache["GCache_complex_index"], dtype=jnp.int32
            ).reshape(-1)
            - 1,
            clm=jnp.asarray(cache["CLMCache3D"], dtype=jnp.complex128),
        )
    return _BatchZMRuntime(
        param=param,
        residue_box=z.get_residue_gaussian_density_cache(param),
        rotation_cache=rotation_cache,
        bbox_to_zm_cache=bbox_to_zm_cache,
        x64_bbox_to_zm_cache=x64_bbox_to_zm_cache,
        descriptor_cache=z.prepare_descriptor_assembly_cache(max_order),
    )


def _descriptor_width(
    mode: int, max_target_order: int, cache: z.DescriptorAssemblyCache
) -> int:
    mean_block_count = max_target_order - 1 if mode in (0, 2) else 0
    width = mean_block_count * int(cache.moment_indices.shape[0])
    if mode in (1, 2):
        width += int(cache.descriptor_indices.shape[0])
    return width


def _prepare_descriptor_runner(
    *,
    max_order: int,
    max_target_order: int,
    mode: int,
    runtime: _BatchZMRuntime,
):
    """Compile the complete device pipeline as one reusable batch executable."""
    return jax.jit(
        partial(
            calculate_descriptor_batch_from_voxels,
            max_order=max_order,
            max_target_order=max_target_order,
            mode=mode,
            default_radius_multiplier=runtime.param["default_radius_multiplier"],
            bbox_to_zm_cache=runtime.bbox_to_zm_cache,
            x64_bbox_to_zm_cache=runtime.x64_bbox_to_zm_cache,
            rotation_cache=runtime.rotation_cache,
            descriptor_cache=runtime.descriptor_cache,
        )
    )


def _run_prepared_batch(
    paths: Sequence[str],
    *,
    grid_width: float,
    max_order: int,
    max_target_order: int,
    mode: int,
    batch_size: int,
    runtime: _BatchZMRuntime,
    descriptor_runner=None,
    prefetch: bool = False,
) -> z.DescriptorVector:
    if not paths:
        width = _descriptor_width(mode, max_target_order, runtime.descriptor_cache)
        return z.DescriptorVector(
            values=jnp.empty((0, width), dtype=z.FLOAT_DTYPE),
            is_valid=jnp.empty((0, width), dtype=bool),
        )

    if descriptor_runner is None:
        descriptor_runner = _prepare_descriptor_runner(
            max_order=max_order,
            max_target_order=max_target_order,
            mode=mode,
            runtime=runtime,
        )

    def prepare_chunk(chunk_paths: Sequence[str]) -> np.ndarray:
        host_voxels = []
        for path in chunk_paths:
            xyz, residues = z.get_pdb_xyz_ca(path)
            voxel, _corner = fill_voxel_by_weight_density_host(
                xyz,
                residues,
                runtime.param["residue_weight_map"],
                grid_width,
                runtime.residue_box[grid_width],
            )
            host_voxels.append(voxel)
        return pad_voxel_batch(host_voxels)

    path_chunks = [
        paths[start : start + batch_size]
        for start in range(0, len(paths), batch_size)
    ]
    chunks: list[z.DescriptorVector] = []
    if not prefetch or len(path_chunks) == 1:
        prepared_chunks = map(prepare_chunk, path_chunks)
        for prepared in prepared_chunks:
            chunks.append(descriptor_runner(jnp.asarray(prepared, dtype=z.FLOAT_DTYPE)))
    else:
        # At most two padded host chunks are pending. Results are consumed in
        # submission order, so parsing completion cannot reorder descriptors.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="zmpy3d-prepare") as pool:
            pending: list[Future[np.ndarray]] = []
            next_index = 0
            while next_index < min(2, len(path_chunks)):
                pending.append(pool.submit(prepare_chunk, path_chunks[next_index]))
                next_index += 1
            while pending:
                prepared = pending.pop(0).result()
                if next_index < len(path_chunks):
                    pending.append(pool.submit(prepare_chunk, path_chunks[next_index]))
                    next_index += 1
                chunks.append(
                    descriptor_runner(jnp.asarray(prepared, dtype=z.FLOAT_DTYPE))
                )

    return z.DescriptorVector(
        values=jnp.concatenate([chunk.values for chunk in chunks], axis=0),
        is_valid=jnp.concatenate([chunk.is_valid for chunk in chunks], axis=0),
    )


def ZMPY3D_CLI_BatchZM(
    PDBFileName: Sequence[str],
    GridWidth: float = 1.0,
    MaxOrder: int = 6,
    MaxTargetOrder2NormRotate: int = 5,
    Mode: int = 0,
    BatchSize: int = 16,
) -> z.DescriptorVector:
    """
    Calculate 3D Zernike moments for a batch of PDB structures.

    This function processes multiple PDB files and computes rotation-invariant 3D Zernike
    moment descriptors for each structure. It supports different normalization schemes
    and can handle various grid widths and maximum orders.

    Parameters
    ----------
    PDBFileName : list of str
        List of paths to PDB files (must end with .pdb or .txt) in old PDB text format.
    GridWidth : float, optional
        Voxel grid width in Angstroms. Must be 0.25, 0.50, or 1.00. Default is 1.0.
    MaxOrder : int, optional
        Maximum order for calculating Zernike moments. Must be 6, 20, or 40. Default is 6.
    MaxTargetOrder2NormRotate : int, optional
        Maximum order for normalization rotation. Must be >= 2 and <= MaxOrder. Default is 5.
    Mode : int, optional
        Descriptor calculation mode:
        - 0: Canterakis normalization only (rotation-invariant moments for orders 2-MaxTargetOrder2NormRotate)
        - 1: 3DZD 121 invariant descriptor only
        - 2: Both Canterakis normalization and 3DZD 121 invariant
        Default is 0.
    BatchSize : int, optional
        Maximum number of structures padded and processed together. Default is 16.

    Returns
    -------
    DescriptorVector
        Stacked JAX descriptor values and masks with shape ``(batch, width)``.

    Notes
    -----
    - This is the batch processing version of ZMPY3D_CLI_ZM.
    - All structures are processed with the same parameters.
    - Pre-cached coefficients are loaded once and reused for all structures.
    - Useful for generating descriptors for shape database searches or clustering.

    Examples
    --------
    >>> pdb_files = ['protein1.pdb', 'protein2.pdb', 'protein3.pdb']
    >>> descriptors = ZMPY3D_CLI_BatchZM(pdb_files, GridWidth=1.0, MaxOrder=20, Mode=2)
    >>> print(f"Computed {descriptors.values.shape[0]} descriptors")
    """
    if Mode not in (0, 1, 2):
        raise ValueError("Mode must be 0, 1, or 2")
    if MaxTargetOrder2NormRotate < 2 or MaxTargetOrder2NormRotate > MaxOrder:
        raise ValueError("MaxTargetOrder2NormRotate must be between 2 and MaxOrder")
    if isinstance(BatchSize, bool) or not isinstance(BatchSize, int) or BatchSize <= 0:
        raise ValueError("BatchSize must be a positive integer")

    runtime = _prepare_batch_runtime(GridWidth, MaxOrder)
    return _run_prepared_batch(
        PDBFileName,
        grid_width=GridWidth,
        max_order=MaxOrder,
        max_target_order=MaxTargetOrder2NormRotate,
        mode=Mode,
        batch_size=BatchSize,
        runtime=runtime,
    )


def main() -> None:
    """
    Main function to execute the ZMPY3D_CLI_BatchZM script from the command line.

    This function parses command line arguments, validates input files, and
    calls the ZMPY3D_CLI_BatchZM function to compute Zernike moment descriptors
    for a list of PDB files. The results are then printed to the standard output.

    Command line arguments:
    - input_file: The input file containing paths to PDB or TXT files.
    - GW: Grid width, must be one of [0.25, 0.50, 1.00].
    - MaxOrder: Maximum order for Zernike moment calculation, must be one of [6, 20, 40].
    - MaxN: Maximum normalization order, must be an integer >= 2 and <= MaxOrder.
    - Mode: Calculation mode, must be one of [0, 1, 2].

    The input file must be a text file with one PDB or TXT file path per line.
    The script will compute the Zernike moments for each structure using the
    specified parameters and print the resulting descriptors to the console.
    """
    parser = argparse.ArgumentParser(
        description="Process a .txt file that contains paths to .pdb or .txt files."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="The input file to process (must end with .txt) containing paths to .pdb or .txt files.",
    )
    parser.add_argument(
        "GW",
        type=float,
        choices=[0.25, 0.50, 1.00],
        help="Grid width must be 0.25, 0.50 or 1.00.",
    )
    parser.add_argument(
        "MaxOrder",
        type=int,
        choices=[6, 20, 40],
        help="Maximum order of calculating ZM must be 6, 20, or 40.",
    )
    parser.add_argument(
        "MaxN", type=int, help="Maximum normalisation order must be an integer number."
    )
    parser.add_argument(
        "Mode", type=int, choices=[0, 1, 2], help="Mode must be 0, 1 or 2."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Maximum structures processed in one padded device batch (default: 16).",
    )

    args = parser.parse_args()

    input_file = args.input_file
    if not input_file.endswith(".txt"):
        parser.error("File must end with .txt")

    if not os.path.isfile(input_file):
        parser.error("File does not exist")

    if args.MaxN > args.MaxOrder or args.MaxN < 2:
        parser.error(
            "Maximum normalisation order must be larger than 2 and less than or equal to the maximum order of calculating ZM."
        )
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")

    with open(input_file, "r") as file:
        lines = file.readlines()

    pdb_file_list = []
    for line in lines:
        pdb_file = line.strip()

        if not (pdb_file.endswith(".pdb") or pdb_file.endswith(".txt")):
            parser.error("File must end with .pdb or .txt")

        if not os.path.isfile(pdb_file):
            parser.error("File does not exist")

        pdb_file_list.append(pdb_file)

    Result = ZMPY3D_CLI_BatchZM(
        pdb_file_list,
        args.GW,
        args.MaxOrder,
        args.MaxN,
        args.Mode,
        BatchSize=args.batch_size,
    )

    np.set_printoptions(
        threshold=Result.values.shape[1] if Result.values.ndim == 2 else 0
    )

    for values, mask in zip(np.asarray(Result.values), np.asarray(Result.is_valid)):
        print(values[mask])
        print("\n")


if __name__ == "__main__":
    main()
