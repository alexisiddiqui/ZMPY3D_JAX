### 1. Zernike Moment Calculation for a Single PDB Structure

This example demonstrates how to calculate 3D Zernike moments for a single PDB structure, including voxelization and descriptor generation.

```python
import ZMPY3D_JAX as z
import numpy as np
import pickle
import os

# Define parameters
MaxOrder = 20
GridWidth = 1.00
Param = z.get_global_parameter()

# Load precalculated cache
# Assuming cache_data is in the same directory as ZMPY3D_JAX package
# Adjust path if necessary based on your installation
cache_dir = os.path.join(os.path.dirname(z.__file__), 'cache_data')
LogCacheFilePath = os.path.join(cache_dir, 'LogG_CLMCache_MaxOrder{:02d}.pkl'.format(MaxOrder))

with open(LogCacheFilePath, 'rb') as file:
    CachePKL = pickle.load(file)

GCache_pqr_linear = np.array(CachePKL['GCache_pqr_linear'])
GCache_complex = np.array(CachePKL['GCache_complex'])
GCache_complex_index = np.array(CachePKL['GCache_complex_index'])
CLMCache3D = np.array(CachePKL['CLMCache3D'], dtype=np.complex128)
CLMCache = np.array(CachePKL['CLMCache'], dtype=np.float64)

# Example PDB file (replace with your actual PDB file path)
PDBFileName = './1WAC_A.txt' # Assuming 1WAC_A.txt is in the current directory

# Convert structure data into coordinates
XYZ, AA_NameList = z.get_pdb_xyz_ca(PDBFileName)

# Convert coordinates into voxels using precalculated Gaussian densities
ResidueBox = z.get_residue_gaussian_density_cache(Param)
Voxel3D, Corner = z.fill_voxel_by_weight_density(XYZ, AA_NameList, Param['residue_weight_map'], GridWidth, ResidueBox[GridWidth])

# Calculate bounding box moments, volume, center, and molecular radius
Dimension_BBox_scaled = np.shape(Voxel3D)
X_sample = np.arange(Dimension_BBox_scaled[0] + 1, dtype=np.float64)
Y_sample = np.arange(Dimension_BBox_scaled[1] + 1, dtype=np.float64)
Z_sample = np.arange(Dimension_BBox_scaled[2] + 1, dtype=np.float64)

XYZ_SampleStruct = {'X_sample': X_sample, 'Y_sample': Y_sample, 'Z_sample': Z_sample}

VolumeMass, Center, _ = z.calculate_bbox_moment(Voxel3D, 1, XYZ_SampleStruct)
AverageVoxelDist2Center, _ = z.calculate_molecular_radius(Voxel3D, Center, VolumeMass, Param['default_radius_multiplier'])
Sphere_X_sample, Sphere_Y_sample, Sphere_Z_sample = z.get_bbox_moment_xyz_sample(Center, AverageVoxelDist2Center, Dimension_BBox_scaled)

SphereXYZ_SampleStruct = {'X_sample': Sphere_X_sample, 'Y_sample': Sphere_Y_sample, 'Z_sample': Sphere_Z_sample}

_, _, SphereBBoxMoment = z.calculate_bbox_moment(Voxel3D, MaxOrder, SphereXYZ_SampleStruct)

# Convert to scaled 3D Zernike moments
ZMoment_scaled, _ = z.calculate_bbox_moment_2_zm(MaxOrder, GCache_complex, GCache_pqr_linear, GCache_complex_index, CLMCache3D, SphereBBoxMoment)

# Convert the scaled 3D Zernike moments into 3DZD-based descriptors
ZM_3DZD_invariant = z.get_3dzd_121_descriptor(ZMoment_scaled)
ZM_3DZD_invariant_121 = ZM_3DZD_invariant[~np.isnan(ZM_3DZD_invariant)]

print("3D Zernike Descriptor (121 invariant):")
print(ZM_3DZD_invariant_121)
```

**CLI Usage:**

```bash
ZMPY3D_JAX_CLI_ZM "./1WAC_A.txt" 1.0 20 2 1
```

### 2. Shape Score Calculation for a Pair of PDB Structures

This example demonstrates how to calculate shape similarity scores (ZMScore and GeoScore) between two PDB structures.

```python
import ZMPY3D_JAX as z
import numpy as np
import pickle
import os

# Define parameters
MaxOrder = 20
GridWidth = 1.00
Param = z.get_global_parameter()

# Load precalculated cache
cache_dir = os.path.join(os.path.dirname(z.__file__), 'cache_data')
BinomialCacheFilePath = os.path.join(cache_dir, 'BinomialCache.pkl')
LogCacheFilePath = os.path.join(cache_dir, 'LogG_CLMCache_MaxOrder{:02d}.pkl'.format(MaxOrder))

with open(BinomialCacheFilePath, 'rb') as file:
    BinomialCachePKL = pickle.load(file)
with open(LogCacheFilePath, 'rb') as file:
    CachePKL = pickle.load(file)

BinomialCache = np.array(BinomialCachePKL['BinomialCache'], dtype=np.float64)

GCache_pqr_linear = np.array(CachePKL['GCache_pqr_linear'], dtype=np.int32)
GCache_complex = np.array(CachePKL['GCache_complex'], dtype=np.complex128)
GCache_complex_index = np.array(CachePKL['GCache_complex_index'], dtype=np.int32)
CLMCache3D = np.array(CachePKL['CLMCache3D'], dtype=np.complex128)
CLMCache = np.array(CachePKL['CLMCache'], dtype=np.float64)

RotationIndex = CachePKL['RotationIndex']
s_id = np.array(np.squeeze(RotationIndex['s_id'][0,0])-1, dtype=np.int32)
n = np.array(np.squeeze(RotationIndex['n'][0,0]), dtype=np.int32)
l = np.array(np.squeeze(RotationIndex['l'][0,0]), dtype=np.int32)
m = np.array(np.squeeze(RotationIndex['m'][0,0]), dtype=np.int32)
mu = np.array(np.squeeze(RotationIndex['mu'][0,0]), dtype=np.int32)
k = np.array(np.squeeze(RotationIndex['k'][0,0]), dtype=np.int32)
IsNLM_Value = np.array(np.squeeze(RotationIndex['IsNLM_Value'][0,0])-1, dtype=np.int32)

# Helper function to get descriptors for a single PDB
def get_descriptors(pdb_filename, MaxOrder, GridWidth, Param, ResidueBox,
                    BinomialCache, GCache_complex, GCache_pqr_linear,
                    GCache_complex_index, CLMCache3D, CLMCache,
                    s_id, n, l, m, mu, k, IsNLM_Value):

    XYZ, AA_NameList = z.get_pdb_xyz_ca(pdb_filename)
    Voxel3D, Corner = z.fill_voxel_by_weight_density(XYZ, AA_NameList, Param['residue_weight_map'], GridWidth, ResidueBox[GridWidth])

    Dimension_BBox_scaled = np.shape(Voxel3D)
    X_sample = np.arange(Dimension_BBox_scaled[0] + 1, dtype=np.float64)
    Y_sample = np.arange(Dimension_BBox_scaled[1] + 1, dtype=np.float64)
    Z_sample = np.arange(Dimension_BBox_scaled[2] + 1, dtype=np.float64)

    XYZ_SampleStruct = {'X_sample': X_sample, 'Y_sample': Y_sample, 'Z_sample': Z_sample}

    VolumeMass, Center, _ = z.calculate_bbox_moment(Voxel3D, 1, XYZ_SampleStruct)
    AverageVoxelDist2Center, _ = z.calculate_molecular_radius(Voxel3D, Center, VolumeMass, Param['default_radius_multiplier'])
    Sphere_X_sample, Sphere_Y_sample, Sphere_Z_sample = z.get_bbox_moment_xyz_sample(Center, AverageVoxelDist2Center, Dimension_BBox_scaled)

    SphereXYZ_SampleStruct = {'X_sample': Sphere_X_sample, 'Y_sample': Sphere_Y_sample, 'Z_sample': Sphere_Z_sample}

    _, _, SphereBBoxMoment = z.calculate_bbox_moment(Voxel3D, MaxOrder, SphereXYZ_SampleStruct)
    ZMoment_scaled, ZMoment_raw = z.calculate_bbox_moment_2_zm(MaxOrder, GCache_complex, GCache_pqr_linear, GCache_complex_index, CLMCache3D, SphereBBoxMoment)

    ZMList = []
    ZM_3DZD_invariant = z.get_3dzd_121_descriptor(ZMoment_scaled)
    ZMList.append(ZM_3DZD_invariant)

    MaxTargetOrder2NormRotate = 5
    for TargetOrder2NormRotate in range(2, MaxTargetOrder2NormRotate + 1):
        ABList = z.calculate_ab_rotation(ZMoment_raw, TargetOrder2NormRotate)
        ZM = z.calculate_zm_by_ab_rotation(ZMoment_raw, BinomialCache, ABList, MaxOrder, CLMCache, s_id, n, l, m, mu, k, IsNLM_Value)
        ZM_mean, _ = z.get_mean_invariant(ZM)
        ZMList.append(ZM_mean)

    MomentInvariant = np.concatenate([val[~np.isnan(val)] for val in ZMList])

    TotalResidueWeight = z.get_total_residue_weight(AA_NameList, Param['residue_weight_map'])
    Prctile_list, STD_XYZ_dist2center, S, K = z.get_ca_distance_info(XYZ)

    GeoDescriptor = np.vstack((AverageVoxelDist2Center, TotalResidueWeight, Prctile_list, STD_XYZ_dist2center, S, K))
    return MomentInvariant, GeoDescriptor

ResidueBox = z.get_residue_gaussian_density_cache(Param)
P = z.get_descriptor_property()

# PDB files for comparison
PDBFileName_A = './1WAC_A.txt'
PDBFileName_B = './2JL9_A.txt'

# Get descriptors for both structures
MomentInvariantRawA, GeoDescriptorA = get_descriptors(PDBFileName_A, MaxOrder, GridWidth, Param, ResidueBox,
                                                     BinomialCache, GCache_complex, GCache_pqr_linear,
                                                     GCache_complex_index, CLMCache3D, CLMCache,
                                                     s_id, n, l, m, mu, k, IsNLM_Value)
MomentInvariantRawB, GeoDescriptorB = get_descriptors(PDBFileName_B, MaxOrder, GridWidth, Param, ResidueBox,
                                                     BinomialCache, GCache_complex, GCache_pqr_linear,
                                                     GCache_complex_index, CLMCache3D, CLMCache,
                                                     s_id, n, l, m, mu, k, IsNLM_Value)

# Predefined weights and indices for Zernike moments and geometric descriptors
ZMIndex = np.vstack([P['ZMIndex0'], P['ZMIndex1'], P['ZMIndex2'], P['ZMIndex3'], P['ZMIndex4']])
ZMWeight = np.vstack([P['ZMWeight0'], P['ZMWeight1'], P['ZMWeight2'], P['ZMWeight3'], P['ZMWeight4']])

# Computing scores
ZMScore = np.sum(np.abs(MomentInvariantRawA[ZMIndex] - MomentInvariantRawB[ZMIndex]) * ZMWeight)
GeoScore = np.sum(np.asarray(P['GeoWeight']) * (2 * np.abs(GeoDescriptorA - GeoDescriptorB) / (1 + np.abs(GeoDescriptorA) + np.abs(GeoDescriptorB))))

# Rescale scores
GeoScoreScaled = (6.6 - GeoScore) / 6.6 * 100.0
ZMScoreScaled = (9.0 - ZMScore) / 9.0 * 100.0

print(f"GeoScore: {GeoScoreScaled:.2f}")
print(f"TotalZMScore: {ZMScoreScaled:.2f}")
```

**CLI Usage:**

```bash
ZMPY3D_JAX_CLI_ShapeScore "./1WAC_A.txt" "./2JL9_A.txt" 1.0
```

### 3. Superimposition of Two PDB Structures

This example demonstrates how to calculate a transformation matrix to superimpose one PDB structure onto another based on their Zernike moment descriptors, and then apply this transformation.

```python
import ZMPY3D_JAX as z
import numpy as np
import pickle
import os

# Define parameters
MaxOrder = 6 # MaxOrder for superimposition is typically lower
GridWidth = 1.0
Param = z.get_global_parameter()
ResidueBox = z.get_residue_gaussian_density_cache(Param)

# Load precalculated cache
cache_dir = os.path.join(os.path.dirname(z.__file__), 'cache_data')
BinomialCacheFilePath = os.path.join(cache_dir, 'BinomialCache.pkl')
LogCacheFilePath = os.path.join(cache_dir, 'LogG_CLMCache_MaxOrder{:02d}.pkl'.format(MaxOrder))

with open(BinomialCacheFilePath, 'rb') as file:
    BinomialCachePKL = pickle.load(file)
with open(LogCacheFilePath, 'rb') as file:
    CachePKL = pickle.load(file)

BinomialCache = BinomialCachePKL['BinomialCache']
GCache_pqr_linear = CachePKL['GCache_pqr_linear']
GCache_complex = CachePKL['GCache_complex']
GCache_complex_index = CachePKL['GCache_complex_index']
CLMCache3D = CachePKL['CLMCache3D']
CLMCache = CachePKL['CLMCache']
RotationIndex = CachePKL['RotationIndex']

s_id = np.squeeze(RotationIndex['s_id'][0,0])-1
n = np.squeeze(RotationIndex['n'][0,0])
l = np.squeeze(RotationIndex['l'][0,0])
m = np.squeeze(RotationIndex['m'][0,0])
mu = np.squeeze(RotationIndex['mu'][0,0])
k = np.squeeze(RotationIndex['k'][0,0])
IsNLM_Value = np.squeeze(RotationIndex['IsNLM_Value'][0,0])-1

# Helper function to get Zernike moment lists for superimposition
def get_zm_for_superimposition(pdb_filename, MaxOrder, GridWidth, Param, ResidueBox,
                                BinomialCache, CLMCache, CLMCache3D, GCache_complex,
                                GCache_complex_index, GCache_pqr_linear, RotationIndex,
                                s_id, n, l, m, mu, k, IsNLM_Value):

    XYZ, AA_NameList = z.get_pdb_xyz_ca(pdb_filename)
    Voxel3D, Corner = z.fill_voxel_by_weight_density(XYZ, AA_NameList, Param['residue_weight_map'], GridWidth, ResidueBox[GridWidth])

    Dimension_BBox_scaled = Voxel3D.shape
    XYZ_SampleStruct = {
        'X_sample': np.arange(Dimension_BBox_scaled[0] + 1),
        'Y_sample': np.arange(Dimension_BBox_scaled[1] + 1),
        'Z_sample': np.arange(Dimension_BBox_scaled[2] + 1)
    }

    VolumeMass, Center, _ = z.calculate_bbox_moment(Voxel3D, 1, XYZ_SampleStruct)
    AverageVoxelDist2Center, _ = z.calculate_molecular_radius(Voxel3D, Center, VolumeMass, Param['default_radius_multiplier'])
    Center_scaled = Center * GridWidth + Corner

    SphereXYZ_SampleStruct = z.get_bbox_moment_xyz_sample(Center, AverageVoxelDist2Center, Dimension_BBox_scaled)
    _, _, SphereBBoxMoment = z.calculate_bbox_moment(Voxel3D, MaxOrder, SphereXYZ_SampleStruct)
    _, ZMoment_raw = z.calculate_bbox_moment_2_zm(MaxOrder, GCache_complex, GCache_pqr_linear, GCache_complex_index, CLMCache3D, SphereBBoxMoment)

    ABList_2 = z.calculate_ab_rotation_all(ZMoment_raw, 2)
    ABList_3 = z.calculate_ab_rotation_all(ZMoment_raw, 3)
    ABList_4 = z.calculate_ab_rotation_all(ZMoment_raw, 4)
    ABList_5 = z.calculate_ab_rotation_all(ZMoment_raw, 5)
    ABList_6 = z.calculate_ab_rotation_all(ZMoment_raw, 6)

    ABList_all = np.vstack(ABList_2 + ABList_3 + ABList_4 + ABList_5 + ABList_6)
    ZMList_all = z.calculate_zm_by_ab_rotation(ZMoment_raw, BinomialCache, ABList_all, MaxOrder, CLMCache, s_id, n, l, m, mu, k, IsNLM_Value)

    ZMList_all = np.stack(ZMList_all, axis=3)
    ZMList_all = np.transpose(ZMList_all, (2, 1, 0, 3))
    ZMList_all = ZMList_all[~np.isnan(ZMList_all)]
    ZMList_all = np.reshape(ZMList_all, (np.int64(ZMList_all.size / 96), 96))

    return Center_scaled, ABList_all, ZMList_all

# PDB files for superimposition
PDBFileName_A = './6NT5.pdb'
PDBFileName_B = './6NT6.pdb'

# Get Zernike moment lists for both structures
Center_scaled_A, ABList_A, ZMList_A = get_zm_for_superimposition(PDBFileName_A, MaxOrder, GridWidth, Param, ResidueBox,
                                                                 BinomialCache, CLMCache, CLMCache3D, GCache_complex,
                                                                 GCache_complex_index, GCache_pqr_linear, RotationIndex,
                                                                 s_id, n, l, m, mu, k, IsNLM_Value)
Center_scaled_B, ABList_B, ZMList_B = get_zm_for_superimposition(PDBFileName_B, MaxOrder, GridWidth, Param, ResidueBox,
                                                                 BinomialCache, CLMCache, CLMCache3D, GCache_complex,
                                                                 GCache_complex_index, GCache_pqr_linear, RotationIndex,
                                                                 s_id, n, l, m, mu, k, IsNLM_Value)

# Compare all Zernike moments and select the maximum value
M = np.abs(ZMList_A.conj().T @ ZMList_B)
MaxValueIndex = np.where(M == np.max(M))
i, j = MaxValueIndex[0][0], MaxValueIndex[1][0]

# Compute the transformation matrix
RotM_A = z.get_transform_matrix_from_ab_list(ABList_A[i, 0], ABList_A[i, 1], Center_scaled_A)
RotM_B = z.get_transform_matrix_from_ab_list(ABList_B[j, 0], ABList_B[j, 1], Center_scaled_B)
TargetRotM = np.linalg.solve(RotM_B, RotM_A)

print("Transformation Matrix (A to B):")
print(TargetRotM)

# Apply the transformation matrix to structure A and save
z.set_pdb_xyz_rot_m_01(PDBFileName_A, TargetRotM, '6NT5_trans.pdb')
print("Transformed 6NT5.pdb saved as 6NT5_trans.pdb")
```

**CLI Usage:**

```bash
ZMPY3D_JAX_CLI_SuperA2B "./6NT5.pdb" "./6NT6.pdb"
```
