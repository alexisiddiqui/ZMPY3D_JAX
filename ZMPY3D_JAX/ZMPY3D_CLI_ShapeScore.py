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
import pickle
import sys
from typing import Tuple

import numpy as np

import ZMPY3D_JAX as z


def ZMPY3D_CLI_ShapeScore(
    PDBFileNameA: str, PDBFileNameB: str, GridWidth: float
) -> Tuple[float, float]:
    """
    Calculate shape similarity scores between two PDB structures.

    This function computes two complementary shape similarity scores:
    1. ZMScore: Based on 3D Zernike moment descriptors (rotation-invariant shape descriptor)
    2. GeoScore: Based on geometric descriptors (size, weight, distance distributions)

    Both scores are scaled to a 0-100 range where higher values indicate greater similarity.

    Parameters
    ----------
    PDBFileNameA : str
        Path to the first PDB file in old PDB text format. Must end with .pdb or .txt.
    PDBFileNameB : str
        Path to the second PDB file in old PDB text format. Must end with .pdb or .txt.
    GridWidth : float
        Voxel grid width in Angstroms. Must be 0.25, 0.50, or 1.00.

    Returns
    -------
    tuple of (float, float)
        GeoScoreScaled : float
            Scaled geometric similarity score (0-100), where 100 indicates identical geometry.
        ZMScoreScaled : float
            Scaled Zernike moment similarity score (0-100), where 100 indicates identical shape.

    Notes
    -----
    - Zernike moments are calculated up to MaxOrder=20.
    - Both 3DZD 121 invariant and rotation-invariant moments (orders 2-5) are used.
    - Geometric descriptors include: molecular radius, total residue weight,
      distance percentiles, standard deviation, skewness, and kurtosis.
    - Scores are scaled using empirically determined thresholds:
      * GeoScore: (6.6 - raw_score) / 6.6 * 100
      * ZMScore: (9.0 - raw_score) / 9.0 * 100

    Examples
    --------
    >>> geo_score, zm_score = ZMPY3D_CLI_ShapeScore('1WAC_A.pdb', '2JL9_A.pdb', 1.0)
    >>> print(f"Geometric similarity: {geo_score:.2f}")
    >>> print(f"Shape similarity: {zm_score:.2f}")
    """

    def ZMCal(
        PDBFileName: str,
        GridWidth: float,
        BinomialCache: object,
        CLMCache: object,
        CLMCache3D: object,
        GCache_complex: object,
        GCache_complex_index: object,
        GCache_pqr_linear: object,
        MaxOrder: int,
        Param: dict,
        ResidueBox: dict,
        RotationIndex: dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        [XYZ, AA_NameList] = z.get_pdb_xyz_ca(PDBFileName)

        [Voxel3D, Corner] = z.fill_voxel_by_weight_density(
            XYZ, AA_NameList, Param["residue_weight_map"], GridWidth, ResidueBox[GridWidth]
        )
        Dimension_BBox_scaled = Voxel3D.shape

        XYZ_SampleStruct = {
            "X_sample": np.arange(Dimension_BBox_scaled[0] + 1),
            "Y_sample": np.arange(Dimension_BBox_scaled[1] + 1),
            "Z_sample": np.arange(Dimension_BBox_scaled[2] + 1),
        }

        [VolumeMass, Center, _] = z.calculate_bbox_moment(Voxel3D, 1, XYZ_SampleStruct)
        [AverageVoxelDist2Center, MaxVoxelDist2Center] = z.calculate_molecular_radius(
            Voxel3D, Center, VolumeMass, Param["default_radius_multiplier"]
        )

        ##################################################################################
        # You may add any preprocessing on the voxel before applying the Zernike moment. #
        ##################################################################################

        SphereXYZ_SampleStruct = z.get_bbox_moment_xyz_sample(
            Center, AverageVoxelDist2Center, Dimension_BBox_scaled
        )
        _, _, SphereBBoxMoment = z.calculate_bbox_moment(Voxel3D, MaxOrder, SphereXYZ_SampleStruct)

        [ZMoment_scaled, ZMoment_raw] = z.calculate_bbox_moment_2_zm(
            MaxOrder,
            GCache_complex,
            GCache_pqr_linear,
            GCache_complex_index,
            CLMCache3D,
            SphereBBoxMoment,
        )
        # ZMoment_scaled[np.isnan(ZMoment_raw)]=np.nan

        ZM_3DZD_invariant = z.get_3dzd_121_descriptor(ZMoment_scaled)

        TargetOrder2NormRotate = 2
        ABList_2 = z.calculate_ab_rotation(ZMoment_raw, TargetOrder2NormRotate)
        ZMList_2 = z.calculate_zm_by_ab_rotation(
            ZMoment_raw,
            BinomialCache,
            ABList_2,
            MaxOrder,
            CLMCache,
            s_id,
            n,
            l,
            m,
            mu,
            k,
            IsNLM_Value,
        )
        [ZM_2, _] = z.get_mean_invariant(ZMList_2)

        TargetOrder2NormRotate = 3
        ABList_3 = z.calculate_ab_rotation(ZMoment_raw, TargetOrder2NormRotate)
        ZMList_3 = z.calculate_zm_by_ab_rotation(
            ZMoment_raw,
            BinomialCache,
            ABList_3,
            MaxOrder,
            CLMCache,
            s_id,
            n,
            l,
            m,
            mu,
            k,
            IsNLM_Value,
        )
        [ZM_3, _] = z.get_mean_invariant(ZMList_3)

        TargetOrder2NormRotate = 4
        ABList_4 = z.calculate_ab_rotation(ZMoment_raw, TargetOrder2NormRotate)
        ZMList_4 = z.calculate_zm_by_ab_rotation(
            ZMoment_raw,
            BinomialCache,
            ABList_4,
            MaxOrder,
            CLMCache,
            s_id,
            n,
            l,
            m,
            mu,
            k,
            IsNLM_Value,
        )
        [ZM_4, _] = z.get_mean_invariant(ZMList_4)

        TargetOrder2NormRotate = 5
        ABList_5 = z.calculate_ab_rotation(ZMoment_raw, TargetOrder2NormRotate)
        ZMList_5 = z.calculate_zm_by_ab_rotation(
            ZMoment_raw,
            BinomialCache,
            ABList_5,
            MaxOrder,
            CLMCache,
            s_id,
            n,
            l,
            m,
            mu,
            k,
            IsNLM_Value,
        )
        [ZM_5, _] = z.get_mean_invariant(ZMList_5)

        MomentInvariant = np.concatenate(
            [z[~np.isnan(z)] for z in [ZM_3DZD_invariant, ZM_2, ZM_3, ZM_4, ZM_5]]
        )

        TotalResidueWeight = z.get_total_residue_weight(AA_NameList, Param["residue_weight_map"])

        [Prctile_list, STD_XYZ_dist2center, S, K] = z.get_ca_distance_info(XYZ)

        GeoDescriptor = np.vstack(
            (AverageVoxelDist2Center, TotalResidueWeight, Prctile_list, STD_XYZ_dist2center, S, K)
        )

        return MomentInvariant, GeoDescriptor

    Param = z.get_global_parameter()

    MaxOrder = 20

    BinomialCacheFilePath = os.path.join(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_data"), "BinomialCache.pkl"
    )
    with open(
        BinomialCacheFilePath, "rb"
    ) as file:  # Used at the entry point, it requires __file__ to identify the package location
        # with open('./cache_data/BinomialCache.pkl', 'rb') as file: # Can be used in ipynb, but not at the entry point.
        BinomialCachePKL = pickle.load(file)

    LogCacheFilePath = os.path.join(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_data"),
        "LogG_CLMCache_MaxOrder{:02d}.pkl".format(MaxOrder),
    )
    with open(
        LogCacheFilePath, "rb"
    ) as file:  # Used at the entry point, it requires __file__ to identify the package location
        # with open('./cache_data/LogG_CLMCache_MaxOrder{:02d}.pkl'.format(MaxOrder), 'rb') as file: # Can be used in ipynb, but not at the entry point.
        CachePKL = pickle.load(file)

    # Extract all cached variables from pickle. These will be converted into a tensor/cupy objects for ZMPY3D_CP and ZMPY3D_TF.
    BinomialCache = BinomialCachePKL["BinomialCache"]

    # GCache, CLMCache, and all RotationIndex
    GCache_pqr_linear = CachePKL["GCache_pqr_linear"]
    GCache_complex = CachePKL["GCache_complex"]
    GCache_complex_index = CachePKL["GCache_complex_index"]
    CLMCache3D = CachePKL["CLMCache3D"]
    CLMCache = CachePKL["CLMCache"]
    RotationIndex = CachePKL["RotationIndex"]

    # RotationIndex is a structure, must be [0,0] to accurately obtain the s_id ... etc, within RotationIndex.
    s_id = np.squeeze(RotationIndex["s_id"][0, 0]) - 1
    n = np.squeeze(RotationIndex["n"][0, 0])
    l = np.squeeze(RotationIndex["l"][0, 0])
    m = np.squeeze(RotationIndex["m"][0, 0])
    mu = np.squeeze(RotationIndex["mu"][0, 0])
    k = np.squeeze(RotationIndex["k"][0, 0])
    IsNLM_Value = np.squeeze(RotationIndex["IsNLM_Value"][0, 0]) - 1

    MaxN = MaxOrder + 1

    ResidueBox = z.get_residue_gaussian_density_cache(Param)

    MomentInvariantRawA, GeoDescriptorA = ZMCal(
        PDBFileNameA,
        GridWidth,
        BinomialCache,
        CLMCache,
        CLMCache3D,
        GCache_complex,
        GCache_complex_index,
        GCache_pqr_linear,
        MaxOrder,
        Param,
        ResidueBox,
        RotationIndex,
    )
    MomentInvariantRawB, GeoDescriptorB = ZMCal(
        PDBFileNameB,
        GridWidth,
        BinomialCache,
        CLMCache,
        CLMCache3D,
        GCache_complex,
        GCache_complex_index,
        GCache_pqr_linear,
        MaxOrder,
        Param,
        ResidueBox,
        RotationIndex,
    )

    P = z.get_descriptor_property()

    ZMIndex = np.concatenate(
        (P["ZMIndex0"], P["ZMIndex1"], P["ZMIndex2"], P["ZMIndex3"], P["ZMIndex4"])
    )
    ZMWeight = np.concatenate(
        (P["ZMWeight0"], P["ZMWeight1"], P["ZMWeight2"], P["ZMWeight3"], P["ZMWeight4"])
    )

    # Calculating ZMScore
    ZMScore = np.sum(np.abs(MomentInvariantRawA[ZMIndex] - MomentInvariantRawB[ZMIndex]) * ZMWeight)

    # Calculating GeoScore
    GeoScore = np.sum(
        P["GeoWeight"]
        * (
            2
            * np.abs(GeoDescriptorA - GeoDescriptorB)
            / (1 + np.abs(GeoDescriptorA) + np.abs(GeoDescriptorB))
        )
    )

    # Calculating paper loss
    Paper_Loss = ZMScore + GeoScore

    # Scaled scores, fitted to shape service
    GeoScoreScaled = (6.6 - GeoScore) / 6.6 * 100.0
    ZMScoreScaled = (9.0 - ZMScore) / 9.0 * 100.0

    return GeoScoreScaled, ZMScoreScaled


def main() -> None:
    """Main function to execute shape similarity scoring from command line arguments.

    This function serves as the entry point for the ZMPY3D_CLI_ShapeScore module when
    executed as a standalone script. It parses command line arguments, validates input
    files and grid width, and invokes the shape scoring computation. The resulting
    similarity scores are then printed to the console.

    Command line arguments:
    - Two input files (PDB or TXT) containing protein structure data.
    - A grid width value (0.25, 0.50, or 1.0) specifying the voxel size for 3D
      grid representation of the structures.

    The function expects exactly three arguments: the first input file, the second input
    file, and the grid width. It will terminate with an error message if the number of
    arguments is incorrect or if the provided files do not exist.

    The computed geometric and Zernike moment similarity scores are scaled to a 0-100
    range and printed as output.
    """
    if len(sys.argv) != 4:
        print("Usage: ZMPY3D_CLI_ShapeScore PDB_A PDB_B GridWidth")
        print(
            "    This function takes two PDB structures and a grid width to generate shape analysis scores."
        )
        print("Error: You must provide exactly two input files and an input grid width.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Process two .pdb or .txt files.")
    parser.add_argument(
        "input_file1",
        type=str,
        help="The first input file to process (must end with .pdb or .txt) with old PDB text format",
    )
    parser.add_argument(
        "input_file2",
        type=str,
        help="The second input file to process (must end with .pdb or .txt) with old PDB text format",
    )
    parser.add_argument(
        "grid_width", type=str, help="The third input file to process (must 0.25, 0.50 or 1.0)"
    )

    args = parser.parse_args()

    input_files = [args.input_file1, args.input_file2]
    for input_file in input_files:
        if not (input_file.endswith(".pdb") or input_file.endswith(".txt")):
            parser.error("File must end with .pdb or .txt")

        if not os.path.isfile(input_file):
            parser.error("File does not exist")

    try:
        GridWidth = float(args.grid_width)
    except ValueError:
        print("GridWidth cannot be converted to a float.")

    if GridWidth not in [0.25, 0.50, 1.0]:
        parser.error("grid width must be either 0.25, 0.50, or 1.0")

    GeoScoreScaled, ZMScoreScaled = ZMPY3D_CLI_ShapeScore(
        args.input_file1, args.input_file2, GridWidth
    )

    print("The scaled score for the geometric descriptor is calculated.")
    print(f"GeoScore {GeoScoreScaled:.2f}")

    print("The scaled score for the Zernike moments is calculated.")
    print(f"TotalZMScore {ZMScoreScaled:.2f}")


if __name__ == "__main__":
    main()
