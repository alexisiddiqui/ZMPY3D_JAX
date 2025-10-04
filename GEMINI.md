# ZMPY3D_JAX Codebase Summary

## Demo Notebooks

*   **ZMPY3D_demo_JAX.ipynb**: Demonstrates the usage of ZMPY3D with JAX, likely showcasing how to leverage JAX for accelerated numerical computations of Zernike moments and related descriptors.
*   **ZMPY3D_demo_shape.ipynb**: Illustrates the process of calculating and comparing molecular shapes using Zernike moments, likely focusing on the `ShapeScore` functionality.
*   **ZMPY3D_demo_super.ipynb**: Provides examples for superimposing molecular structures based on their Zernike moment descriptors, demonstrating the `SuperA2B` functionality.
*   **ZMPY3D_demo_zm.ipynb**: Focuses on the core Zernike moment calculation, showing how to generate Zernike descriptors for individual molecular structures.

## Command Line Interface (CLI) Tools

### `ZMPY3D_JAX/ZMPY3D_CLI_BatchShapeScore.py`
**Function:** Computes shape analysis scores for a batch of paired PDB structures.
**Numerical Flow:**
1.  Loads pre-calculated binomial and CLM caches.
2.  For each pair of PDB files:
    *   Reads PDB coordinates and amino acid names.
    *   Fills a 3D voxel grid with weighted density based on residue types and grid width.
    *   Calculates bounding box moments, volume, center, and molecular radius.
    *   Transforms bounding box moments into Zernike moments (raw and scaled).
    *   Calculates 3DZD invariant descriptor.
    *   Calculates rotation-invariant Zernike moments for orders 2-5 using `calculate_ab_rotation` and `calculate_zm_by_ab_rotation`.
    *   Concatenates all moment invariants.
    *   Calculates geometric descriptors (total residue weight, CA distance info).
    *   Computes ZMScore and GeoScore by comparing descriptors of the two structures.
    *   Scales the scores.

### `ZMPY3D_JAX/ZMPY3D_CLI_BatchSuperA2B.py`
**Function:** Generates transformation matrices for superimposing a batch of paired PDB structures (structure A onto B).
**Numerical Flow:**
1.  Loads pre-calculated binomial and CLM caches.
2.  For each pair of PDB files:
    *   Performs Zernike moment calculation (`ZMCal` function, similar to `BatchShapeScore` but with `MaxOrder=6`).
    *   Calculates `ABList_all` by combining `calculate_ab_rotation_all` for orders 2-6.
    *   Calculates `ZMList_all` using `calculate_zm_by_ab_rotation`.
    *   Computes a similarity matrix `M` by `ZMList_A.conj().T @ ZMList_B`.
    *   Finds the maximum value in `M` to identify the best rotation pair (i, j).
    *   Retrieves transformation matrices `RotM_A` and `RotM_B` using `get_transform_matrix_from_ab_list`.
    *   Calculates the target rotation matrix `TargetRotM = np.linalg.solve(RotM_B, RotM_A)`.

### `ZMPY3D_JAX/ZMPY3D_CLI_BatchZM.py`
**Function:** Computes Zernike moments for a batch of PDB structures based on specified maximum order, normalization order, and voxel gridding width.
**Numerical Flow:**
1.  Loads pre-calculated binomial and CLM caches.
2.  For each PDB file:
    *   Reads PDB coordinates and amino acid names.
    *   Fills a 3D voxel grid with weighted density.
    *   Calculates bounding box moments, volume, center, and molecular radius.
    *   Transforms bounding box moments into Zernike moments (raw and scaled).
    *   Based on `Mode` parameter:
        *   `Mode == 0`: Calculates rotation-invariant Zernike moments for orders 2 to `MaxTargetOrder2NormRotate` using `calculate_ab_rotation` and `calculate_zm_by_ab_rotation`, then takes the mean invariant.
        *   `Mode == 1`: Calculates the 3DZD 121 invariant descriptor.
        *   `Mode == 2`: Calculates both 3DZD 121 invariant and rotation-invariant Zernike moments (as in Mode 0).
    *   Concatenates the resulting Zernike moments.

### `ZMPY3D_JAX/ZMPY3D_CLI_ShapeScore.py`
**Function:** Computes shape analysis scores for a single pair of PDB structures.
**Numerical Flow:**
1.  Loads pre-calculated binomial and CLM caches.
2.  For each of the two PDB files:
    *   Performs Zernike moment calculation (`ZMCal` function, identical to `BatchShapeScore`).
    *   Calculates 3DZD invariant descriptor.
    *   Calculates rotation-invariant Zernike moments for orders 2-5.
    *   Concatenates all moment invariants.
    *   Calculates geometric descriptors.
3.  Computes ZMScore and GeoScore by comparing descriptors of the two structures.
4.  Scales the scores.

### `ZMPY3D_JAX/ZMPY3D_CLI_SuperA2B.py`
**Function:** Generates a transformation matrix to superimpose a single structure A onto structure B.
**Numerical Flow:**
1.  Loads pre-calculated binomial and CLM caches.
2.  For each of the two PDB files:
    *   Performs Zernike moment calculation (`ZMCal` function, identical to `BatchSuperA2B`).
    *   Calculates `ABList_all` by combining `calculate_ab_rotation_all` for orders 2-6.
    *   Calculates `ZMList_all` using `calculate_zm_by_ab_rotation`.
3.  Computes a similarity matrix `M` by `ZMList_A.conj().T @ ZMList_B`.
4.  Finds the maximum value in `M` to identify the best rotation pair (i, j).
5.  Retrieves transformation matrices `RotM_A` and `RotM_B` using `get_transform_matrix_from_ab_list`.
6.  Calculates the target rotation matrix `TargetRotM = np.linalg.solve(RotM_B, RotM_A)`.

### `ZMPY3D_JAX/ZMPY3D_CLI_ZM.py`
**Function:** Computes Zernike moments for a single PDB structure based on specified maximum order, normalization order, and voxel gridding width.
**Numerical Flow:**
1.  Loads pre-calculated binomial and CLM caches.
2.  Reads PDB coordinates and amino acid names.
3.  Fills a 3D voxel grid with weighted density.
4.  Calculates bounding box moments, volume, center, and molecular radius.
5.  Transforms bounding box moments into Zernike moments (raw and scaled).
6.  Based on `Mode` parameter (identical logic to `ZMPY3D_CLI_BatchZM.py`):
    *   `Mode == 0`: Calculates rotation-invariant Zernike moments for orders 2 to `MaxTargetOrder2NormRotate`.
    *   `Mode == 1`: Calculates the 3DZD 121 invariant descriptor.
    *   `Mode == 2`: Calculates both.
7.  Concatenates the resulting Zernike moments.

## ZMPY3D_JAX/lib/ Python Files Overview

### `ZMPY3D_JAX/lib/write_string_list_as_file.py`
*   **Function:** Writes a list of strings to a specified file, with each string on a new line.
*   **Numerical Flow (JAX Context):** This function is a basic file I/O operation and does not involve numerical computations. It would remain largely unchanged in a JAX conversion, as JAX focuses on numerical operations and not file system interactions.

### `ZMPY3D_JAX/lib/set_pdb_xyz_rot_m_01.py`
*   **Function:** Applies a 3D rotation matrix to the atomic coordinates (X, Y, Z) within a PDB file and writes the transformed coordinates to a new PDB file.
*   **Numerical Flow (JAX Context):**
    1.  Reads PDB file content as a list of strings. This involves file I/O and string parsing, which are outside JAX's direct numerical computation graph.
    2.  Extracts X, Y, Z coordinates for 'ATOM' lines. This involves string parsing and type conversion to floats.
    3.  Constructs a `target_xyz` NumPy array.
    4.  Performs matrix multiplication: `rot_m @ np.hstack((target_xyz, np.ones((target_xyz.shape[0], 1)))).T`. This is a core numerical operation that can be efficiently handled by JAX's `jax.numpy.matmul` or `@` operator.
    5.  Rounds the transformed coordinates. JAX's `jax.numpy.round` can be used.
    6.  Formats the new coordinates back into strings. This involves string formatting and would remain outside JAX's direct numerical computation graph.
    7.  Updates the PDB content and writes to a new file. File I/O remains outside JAX.
    *   **JAX Conversion Notes:** The matrix multiplication and rounding are prime candidates for JAX transformation (e.g., `jit`, `grad`). The string parsing and formatting parts would remain standard Python/NumPy operations.

### `ZMPY3D_JAX/lib/read_file_as_string_list.py`
*   **Function:** Reads a file and returns its content as a list of strings, with each line as an element and stripped of leading/trailing whitespace.
*   **Numerical Flow (JAX Context):** This is a basic file I/O operation and does not involve numerical computations. It would remain largely unchanged in a JAX conversion.

### `ZMPY3D_JAX/lib/get_transform_matrix_from_ab_list02.py`
*   **Function:** Constructs a 4x4 transformation matrix (including rotation and translation) from complex `a` and `b` coefficients and a `center_scaled` vector. This matrix is used for superimposition.
*   **Numerical Flow (JAX Context):**
    1.  Calculates `a2pb2` and `a2mb2` from complex `a` and `b`. These are element-wise complex arithmetic operations. JAX supports complex numbers and operations.
    2.  Constructs `m33_linear` array using real and imaginary parts of complex numbers and products. All these operations are numerically intensive and can be JAX-transformed.
    3.  Reshapes `m33_linear` into a 3x3 rotation matrix `m33`. JAX's `jax.numpy.reshape` can be used.
    4.  Constructs a 4x4 homogeneous transformation matrix `m44` by embedding `m33` and `center_scaled`. This involves array manipulation.
    5.  Calculates the inverse of `m44` using `np.linalg.inv`. This is a critical numerical linear algebra operation that JAX can accelerate with `jax.numpy.linalg.inv`.
    *   **JAX Conversion Notes:** This function is highly amenable to JAX conversion. All NumPy operations involving complex numbers, array reshaping, and linear algebra can be directly replaced with their `jax.numpy` equivalents for potential performance gains and automatic differentiation.

### `ZMPY3D_JAX/lib/get_total_residue_weight.py`
*   **Function:** Calculates the total residue weight of a protein given a list of amino acid names and a mapping of amino acid names to their weights. Defaults to 'ASP' if an amino acid name is not found in the map.
*   **Numerical Flow (JAX Context):**
    1.  Iterates through a list of amino acid names.
    2.  For each amino acid, it looks up its weight in a dictionary (`residue_weight_map`).
    3.  Sums up the weights.
    *   **JAX Conversion Notes:** This function primarily involves dictionary lookups and a loop. While JAX can handle loops, it's generally more efficient to vectorize operations. If `aa_name_list` and `residue_weight_map` can be represented as JAX arrays (e.g., one-hot encoded amino acid names and a weight array), the summation could be vectorized using `jax.numpy.sum` and `jax.numpy.take` or similar operations for efficient JAX compilation. The dictionary lookup for missing amino acids would need careful handling in a JAX-compatible way (e.g., using `jax.lax.select` with a boolean mask).

### `ZMPY3D_JAX/lib/get_residue_weight_map01.py`
*   **Function:** Returns a dictionary mapping three-letter amino acid codes (and some nucleotide codes) to their corresponding residue weights.
*   **Numerical Flow (JAX Context):** This function defines a static dictionary. It does not involve any numerical computations. In a JAX context, this dictionary would likely be converted into a static JAX array or a lookup table if its values are to be used within JAX-transformed functions.

### `ZMPY3D_JAX/lib/get_residue_radius_map01.py`
*   **Function:** Returns a dictionary mapping three-letter amino acid codes (and some nucleotide codes) to their corresponding residue radii, scaled by a constant factor `sqrt(5/3)`.
*   **Numerical Flow (JAX Context):** This function defines a static dictionary and performs a simple scalar multiplication (`math.sqrt(5.0 / 3.0) * value`). In a JAX context, similar to `get_residue_weight_map01`, this dictionary would likely be converted into a static JAX array or a lookup table if its values are to be used within JAX-transformed functions. The `math.sqrt` operation is a standard numerical operation.

### `ZMPY3D_JAX/lib/get_residue_gaussian_density_cache02.py`
*   **Function:** Generates a cache of 3D Gaussian density boxes for different amino acid residues and grid widths. This pre-calculates the density distributions used for voxelization.
*   **Numerical Flow (JAX Context):**
    1.  Calls `calculate_box_by_grid_width` to get initial Gaussian density boxes for a grid width of 1.00. This function itself will need to be JAX-compatible.
    2.  Calculates the sum of densities for each box (`np.sum`). This is a reduction operation that JAX can handle with `jax.numpy.sum`.
    3.  Iterates through different grid widths:
        *   Calls `calculate_box_by_grid_width` again for the current grid width.
        *   Calculates `temp_total_gaussian_density` (sum of densities).
        *   Calculates `density_scalar` which involves division and exponentiation (`gw**3`). These are element-wise numerical operations.
        *   Adjusts the density boxes by multiplying with `density_scalar`. This is element-wise array multiplication.
    4.  Stores these adjusted boxes in dictionaries.
    *   **JAX Conversion Notes:** The core numerical operations (summation, division, exponentiation, array multiplication) are all JAX-compatible. The loops can be handled by `jax.lax.scan` or by ensuring that `calculate_box_by_grid_width` is JAX-transformed and then mapping over the `doable_grid_width_list` if possible. The dictionary creation would remain outside the JAX-transformed functions, but the arrays within them would be JAX arrays.

### `ZMPY3D_JAX/lib/get_pdb_xyz_ca02.py`
*   **Function:** Parses a PDB file to extract the XYZ coordinates and amino acid names specifically for C-alpha (CA) atoms. It also checks for NaN values in coordinates.
*   **Numerical Flow (JAX Context):**
    1.  Reads the PDB file line by line. (File I/O, outside JAX).
    2.  Filters lines to get 'ATOM' records and then further filters for 'CA' atoms. (String operations, outside JAX).
    3.  Extracts X, Y, Z coordinates and amino acid names by string slicing and conversion to float. (String operations, type conversion, outside JAX).
    4.  Appends coordinates and names to lists.
    5.  Checks for NaN values in coordinates using `math.isnan`. This is a numerical check.
    6.  Converts the list of XYZ tuples into a NumPy array.
    *   **JAX Conversion Notes:** The primary numerical part is the creation of the `xyz_matrix` NumPy array. If the parsing and filtering steps can be made JAX-compatible (e.g., by operating on string arrays or pre-parsed numerical data), then the `xyz_matrix` could directly be a JAX array. The NaN check could use `jax.numpy.isnan`. However, the string parsing itself is not directly JAX-transformable and would likely remain a Python/NumPy preprocessing step.

### `ZMPY3D_JAX/lib/get_mean_invariant03.py`
*   **Function:** Calculates the mean and standard deviation of a list of Zernike moment arrays, typically representing different rotations of a molecule.
*   **Numerical Flow (JAX Context):**
    1.  Stacks a list of Zernike moment arrays into a single NumPy array, then takes the absolute value. This involves array stacking (`np.stack`) and element-wise absolute value (`np.abs`).
    2.  Calculates the mean along a specified axis (`np.mean`).
    3.  Calculates the standard deviation along a specified axis (`np.std`) with a delta degrees of freedom of 1.
    *   **JAX Conversion Notes:** All operations (`np.stack`, `np.abs`, `np.mean`, `np.std`) have direct equivalents in `jax.numpy` and are highly suitable for JAX transformation. This function would be very efficient under JAX.

### `ZMPY3D_JAX/lib/get_global_parameter02.py`
*   **Function:** Initializes and returns a dictionary containing various global parameters used throughout the ZMPY3D codebase, such as grid widths, radius multipliers, density calculation constants, and mappings for residue weights and radii.
*   **Numerical Flow (JAX Context):**
    1.  Defines several scalar constants (`sd_cutoff`, `default_radius_multiplier`, `density_multiplier`).
    2.  Calculates `three_over_2pi_32` using `math.pi` and exponentiation. This is a simple numerical calculation.
    3.  Calls `get_residue_weight_map01()` and `get_residue_radius_map01()` to populate parts of the parameter dictionary.
    *   **JAX Conversion Notes:** This function primarily sets up static parameters. The numerical calculation for `three_over_2pi_32` can be done using `jax.numpy.pi` and `jax.numpy.power`. The dictionaries themselves would remain Python dictionaries, but their values (weights, radii) could be converted to JAX arrays if they are to be used in JAX-transformed functions.

### `ZMPY3D_JAX/lib/get_descriptor_property.py`
*   **Function:** Returns a dictionary (`property_struct`) containing various thresholds and pre-defined weights and indices for Zernike moments and geometric descriptors. These are used in calculating shape scores.
*   **Numerical Flow (JAX Context):** This function primarily defines static numerical arrays (`np.array`) and scalar values.
    *   **JAX Conversion Notes:** The NumPy arrays (`GeoWeight`, `ZMIndex0` through `ZMIndex4`, `ZMWeight0` through `ZMWeight4`) can be directly converted to JAX arrays (`jax.numpy.array`). The scalar values would remain as Python floats. This function would mostly serve to initialize JAX-compatible constants.

### `ZMPY3D_JAX/lib/get_ca_distance_info.py`
*   **Function:** Calculates various geometric descriptors from C-alpha (CA) atom coordinates, including percentiles of distances to the center, standard deviation of distances, skewness (s), and kurtosis (k).
*   **Numerical Flow (JAX Context):**
    1.  Calculates the center of mass of the `xyz` coordinates (`np.mean`).
    2.  Calculates the Euclidean distance of each atom to the center (`np.sqrt`, `np.sum`, `**2`).
    3.  Computes percentiles of these distances (`np.percentile`).
    4.  Calculates the standard deviation of distances (`np.std`).
    5.  Calculates skewness (`s`) and kurtosis (`k`) using formulas involving sums, powers, and divisions of normalized distances.
    *   **JAX Conversion Notes:** All NumPy operations (`np.mean`, `np.sqrt`, `np.sum`, `**`, `np.percentile`, `np.std`) have direct equivalents in `jax.numpy` and are highly suitable for JAX transformation. This function would benefit significantly from JAX's numerical acceleration and automatic differentiation capabilities.

### `ZMPY3D_JAX/lib/get_bbox_moment_xyz_sample01.py`
*   **Function:** Generates normalized sample coordinates (X, Y, Z) for a bounding box, centered at a given point and scaled by a radius. These samples are used for calculating Zernike moments.
*   **Numerical Flow (JAX Context):**
    1.  Extracts dimensions from `dimension_bbox_scaled`.
    2.  Generates `x_sample`, `y_sample`, `z_sample` arrays using `np.arange`, subtraction, and division. These are element-wise array operations.
    *   **JAX Conversion Notes:** All NumPy operations (`np.arange`, subtraction, division) have direct equivalents in `jax.numpy` and are highly suitable for JAX transformation. This function would be very efficient under JAX.

### `ZMPY3D_JAX/lib/get_3dzd_121_descriptor02.py`
*   **Function:** Calculates the 3D Zernike Descriptor (3DZD) 121 invariant from scaled Zernike moments. This invariant is a rotation-invariant descriptor of molecular shape.
*   **Numerical Flow (JAX Context):**
    1.  Replaces NaN values in `z_moment_scaled` with 0. This can be done using `jax.numpy.nan_to_num` or `jax.numpy.where` with `jax.numpy.isnan`.
    2.  Calculates the squared absolute value of `z_moment_scaled`. This involves element-wise absolute value and exponentiation (`np.abs`, `**2`).
    3.  Sums along a specified axis to get `z_moment_scaled_norm_positive`. This is a reduction operation (`np.sum`).
    4.  Sets a slice of `z_moment_scaled_norm` to 0, then sums again to get `z_moment_scaled_norm_negative`. This involves array slicing and assignment, and another reduction.
    5.  Calculates the final invariant using `np.sqrt` and addition.
    6.  Replaces small values with NaN. This involves a conditional assignment.
    *   **JAX Conversion Notes:** All numerical operations (`np.abs`, `**`, `np.sum`, `np.sqrt`) have direct equivalents in `jax.numpy`. Array slicing and conditional assignments can be handled with `jax.numpy.where` or `jax.lax.select`. This function is highly suitable for JAX transformation.

### `ZMPY3D_JAX/lib/fill_voxel_by_weight_density04.py`
*   **Function:** Fills a 3D voxel grid with density values based on atomic coordinates, amino acid types, and pre-calculated residue density boxes. This effectively converts a discrete atomic structure into a continuous density map.
*   **Numerical Flow (JAX Context):**
    1.  Calculates the minimum and maximum bounding box points, and the unscaled dimensions (`np.min`, `np.max`, subtraction).
    2.  Determines `max_box_edge` from `residue_box` values.
    3.  Calculates `dimension_bbox_scaled` using `np.ceil`, division, addition, and type casting.
    4.  Calculates `corner_xyz` using subtraction and multiplication.
    5.  Initializes an empty 3D NumPy array `voxel3d`.
    6.  **Loop over atoms:** This is the most computationally intensive part.
        *   Retrieves `aa_name` and handles missing names (dictionary lookup).
        *   Gets `coord` and `aa_box`.
        *   Calculates `coord_box_corner` using `np.fix`, `np.round`, subtraction, division, and type casting.
        *   Defines `start` and `end` for slicing.
        *   **Voxel update:** `voxel3d[start[0]:end[0], start[1]:end[1], start[2]:end[2]] += aa_box * weight_multiplier`. This involves array slicing, element-wise multiplication, and in-place addition.
    *   **JAX Conversion Notes:**
        *   The initial bounding box calculations are straightforward to convert to JAX.
        *   The loop over `num_of_atom` is a challenge for direct JAX transformation, as JAX prefers vectorized operations. This loop would ideally be replaced with `jax.lax.scan` or a custom JAX primitive if the operations within the loop can be made fully JAX-compatible.
        *   The dictionary lookups for `aa_name` and `residue_box` would need to be handled carefully, potentially by converting them to JAX arrays and using `jax.numpy.take` or similar indexing operations.
        *   The array slicing and in-place addition on `voxel3d` are also areas that need careful JAX conversion, as JAX arrays are immutable. This would likely involve `jax.lax.dynamic_update_slice` or similar functional updates.

### `ZMPY3D_JAX/lib/eigen_root.py`
*   **Function:** Calculates the roots of a polynomial given its coefficients by constructing a companion matrix and finding its eigenvalues.
*   **Numerical Flow (JAX Context):**
    1.  Reshapes the input `poly_coefficient_list` into a 1D array.
    2.  Constructs a companion matrix `m` using `np.diag` and array assignment. This involves array creation and manipulation.
    3.  Calculates the eigenvalues of `m` using `np.linalg.eigvals`. This is a core numerical linear algebra operation.
    *   **JAX Conversion Notes:** All NumPy operations (`np.reshape`, `np.diag`, array assignment, `np.linalg.eigvals`) have direct equivalents in `jax.numpy` and `jax.numpy.linalg`. This function is highly suitable for JAX transformation and would benefit from JAX's optimized linear algebra routines.

### `ZMPY3D_JAX/lib/calculate_zm_by_ab_rotation01.py`
*   **Function:** Calculates rotated Zernike moments based on raw Zernike moments and rotation coefficients (`a`, `b`). It uses pre-computed binomial and CLM (Clebsch-Gordan coefficients) caches.
*   **Numerical Flow (JAX Context):**
    1.  Flattens `a` and `b` from `ab_list`.
    2.  Calculates `aac`, `bbc`, `bbcaac`, `abc`, `ab` using complex arithmetic, real part extraction, and division. These are element-wise complex operations.
    3.  Calculates logarithmic powers of these terms (`bbcaac_pow_k_list`, `aac_pow_l_list`, `ab_pow_m_list`, `abc_pow_mu_list`). This involves `np.log` and element-wise multiplication.
    4.  Constructs `f_exp` by conditionally selecting and conjugating `z_moment_raw` elements based on `mu` and its parity, then takes the logarithm. This involves array indexing, conditional logic, and complex arithmetic.
    5.  Retrieves `clm` and `bin` from caches.
    6.  **Loop over `zm_rotated_list` (rotations):** This loop is a critical part for JAX conversion.
        *   Extracts specific elements from the pre-calculated logarithmic power lists.
        *   Calculates `nlm` by summing several logarithmic terms and `f_exp`, `clm`, `bin`. This is element-wise complex addition.
        *   Initializes `z_nlm` and performs an "add at" operation: `np.add.at(z_nlm, s_id, np.exp(nlm))`. This is an in-place update, which is problematic for JAX's immutability.
        *   Reshapes `zm` and transposes it.
    *   **JAX Conversion Notes:**
        *   The initial complex arithmetic and logarithmic power calculations are highly suitable for JAX.
        *   The conditional logic for `f_exp` can be handled with `jax.numpy.where`.
        *   The loop over `zm_rotated_list` needs to be vectorized or transformed using `jax.lax.scan` or `jax.vmap`.
        *   The `np.add.at` operation is a major challenge. In JAX, this would typically be replaced by `jax.ops.index_add` or by re-thinking the accumulation as a functional update, possibly using `jax.scipy.special.logsumexp` if the sum is over exponentials.
        *   Array reshaping and transposing are directly supported by `jax.numpy`.

### `ZMPY3D_JAX/lib/calculate_molecular_radius03.py`
*   **Function:** Calculates the average and maximum molecular radii from a 3D voxel density map, given the center of mass, total volume/mass, and a default radius multiplier.
*   **Numerical Flow (JAX Context):**
    1.  Creates a boolean mask `has_weight` for non-zero voxel elements.
    2.  Extracts `voxel_list` (non-zero densities) and `x_coord`, `y_coord`, `z_coord` (coordinates of non-zero voxels) using boolean indexing and `np.where`.
    3.  Stacks coordinates into `voxel_list_xyz`.
    4.  Calculates squared distances of voxels to the center (`np.sum`, `**2`).
    5.  Calculates `average_voxel_mass2center_squared` using element-wise multiplication, summation, and division.
    6.  Calculates `average_voxel_dist2center` and `max_voxel_dist2center` using `np.sqrt`, multiplication, and `np.max`.
    *   **JAX Conversion Notes:** All NumPy operations (`np.where`, boolean indexing, `np.stack`, `np.sum`, `**`, `np.sqrt`, `np.max`) have direct equivalents in `jax.numpy` and are highly suitable for JAX transformation. This function would be very efficient under JAX.

### `ZMPY3D_JAX/lib/calculate_box_by_grid_width.py`
*   **Function:** Generates 3D Gaussian density boxes for each amino acid residue at a specified grid width. These boxes represent the spatial density distribution of each residue.
*   **Numerical Flow (JAX Context):**
    1.  Retrieves parameters from the `param` dictionary.
    2.  **Loop over `residue_name`:** This loop is a key area for JAX optimization.
        *   Retrieves `weight` and `radius` from maps.
        *   Calculates `sigma`, `box_edge`, `center`, `sqr_radius` using `np.sqrt`, `**`, `np.ceil`, integer division. These are scalar numerical operations.
        *   Generates 3D coordinate grids `x, y, z` using `np.mgrid`. This is a grid generation operation.
        *   Calculates `is_within_radius` using element-wise comparisons and arithmetic.
        *   Calculates `x_sx` and `gaus_val` using complex arithmetic involving `**`, division, multiplication, and `np.exp`. These are element-wise array operations.
        *   Initializes `residue_unit_box` as a zero array.
        *   Conditionally assigns `gaus_val` to `residue_unit_box` based on `is_within_radius`. This involves boolean indexing and assignment.
    *   **JAX Conversion Notes:**
        *   The loop over `residue_name` should be vectorized using `jax.vmap` if possible, or handled with `jax.lax.scan`.
        *   `np.mgrid` can be replaced with `jax.numpy.mgrid`.
        *   All numerical operations (`np.sqrt`, `**`, division, multiplication, `np.exp`, comparisons) have direct `jax.numpy` equivalents.
        *   Boolean indexing and conditional assignment can be handled with `jax.numpy.where` or `jax.lax.select` to maintain immutability.

### `ZMPY3D_JAX/lib/calculate_bbox_moment_2_zm05.py`
*   **Function:** Converts raw bounding box moments into Zernike moments (both raw and scaled). This is a core step in the Zernike moment calculation pipeline.
*   **Numerical Flow (JAX Context):**
    1.  Defines a helper function `complex_nan` to create a complex NaN.
    2.  Reshapes and transposes `bbox_moment`. This involves array manipulation.
    3.  Calculates `zm_geo` by element-wise multiplication of `g_cache_complex` with elements from `bbox_moment` indexed by `g_cache_pqr_linear`. This involves array indexing and complex multiplication.
    4.  Initializes `zm_geo_sum` as a zero array.
    5.  Performs an "add at" operation: `np.add.at(zm_geo_sum, g_cache_complex_index - 1, zm_geo)`. This is an in-place update.
    6.  Replaces zero values in `zm_geo_sum` with complex NaNs. This involves conditional assignment.
    7.  Calculates `z_moment_raw` by multiplying `zm_geo_sum` with a constant.
    8.  Reshapes and transposes `z_moment_raw`.
    9.  Calculates `z_moment_scaled` by element-wise multiplication with `clm_cache3d`.
    *   **JAX Conversion Notes:**
        *   Array reshaping and transposing are directly supported by `jax.numpy`.
        *   Element-wise complex multiplication and array indexing are JAX-compatible.
        *   The `np.add.at` operation is a major challenge for JAX due to immutability. It would need to be replaced by `jax.ops.index_add` or a functional equivalent.
        *   Conditional assignment for NaNs can be handled with `jax.numpy.where`.
        *   All other numerical operations are directly supported by `jax.numpy`.

### `ZMPY3D_JAX/lib/calculate_bbox_moment06.py`
*   **Function:** Calculates 3D bounding box moments up to a specified maximum order from a voxel density map. It uses `tensordot` for efficient computation.
*   **Numerical Flow (JAX Context):**
    1.  Creates an `extend_voxel3d` array by padding `voxel3d` with zeros. This involves array creation and slicing.
    2.  Calculates `diff_extend_voxel3d` by applying `np.diff` three times along different axes. This is a differencing operation.
    3.  Calculates `x_power`, `y_power`, `z_power` arrays using `np.power` and `np.arange`. These are element-wise power calculations and array generation.
    4.  **Core Moment Calculation:** Uses nested `np.tensordot` calls to compute `bbox_moment`. This is a highly optimized tensor contraction operation.
    5.  Generates `p, q, r` grids using `np.meshgrid`.
    6.  Normalizes and transposes `bbox_moment` by dividing by `p, q, r` and transposing.
    7.  Extracts `volume_mass` and `center` from `bbox_moment`.
    *   **JAX Conversion Notes:**
        *   Array creation, slicing, and `np.diff` have direct `jax.numpy` equivalents.
        *   `np.power` and `np.arange` are directly supported.
        *   `np.tensordot` is available in `jax.numpy` and is a prime candidate for JAX acceleration.
        *   `np.meshgrid` is available in `jax.numpy`.
        *   Array transposition and element-wise division are directly supported.
        *   This function is highly amenable to JAX transformation and would be very efficient.

### `ZMPY3D_JAX/lib/calculate_ab_rotation_02_all.py`
*   **Function:** Calculates all possible `a` and `b` rotation coefficients for a given raw Zernike moment array, considering multiple `ind_real` values (orders). It reuses the `eigen_root` function.
*   **Numerical Flow (JAX Context):**
    1.  Determines `abconj_coef` based on `target_order2_norm_rotate` (even or odd) and calls `eigen_root` to find `abconj_sol`. This involves array indexing, conditional logic, complex conjugation, and polynomial root finding.
    2.  Defines a nested function `get_ab_list_by_ind_real`:
        *   Calculates various powers and products of real and imaginary parts of `abconj_sol`. These are element-wise numerical operations.
        *   Constructs `coef4` and `coef3` using complex arithmetic and powers.
        *   Forms `bimbre_coef` array.
        *   Calls `eigen_root` for each row of `bimbre_coef` to find `bimbre_sol_real`. This is a loop over polynomial root finding.
        *   Filters `bimbre_sol_real` based on absolute value.
        *   **Loop over `bimbre_sol_real`:**
            *   Calculates `bre` and `bim` using `np.sqrt`, division, and `np.power`.
            *   Constructs complex `b` and `a` using `np.vectorize(complex)` and complex conjugation.
            *   Appends `a` and `b` to lists.
        *   Concatenates `a_list` and `b_list` into `ab_list`.
    3.  Iterates through `ind_real_all` (orders) and calls `get_ab_list_by_ind_real` for each.
    *   **JAX Conversion Notes:**
        *   The initial `abconj_coef` calculation and `eigen_root` call are JAX-compatible.
        *   Inside `get_ab_list_by_ind_real`:
            *   All element-wise numerical operations and power calculations are JAX-compatible.
            *   The loop for `bimbre_sol_real` and the subsequent loop for `a_list`/`b_list` would need to be vectorized using `jax.vmap` or `jax.lax.scan`.
            *   `np.vectorize(complex)` can be replaced by direct JAX complex number construction.
            *   `np.concatenate` is available in `jax.numpy`.
        *   The outer loop over `ind_real_all` would also need to be vectorized or `jax.lax.scan`-ed.
        *   This function is numerically intensive and would greatly benefit from JAX transformation, but requires careful handling of loops and conditional logic.

### `ZMPY3D_JAX/lib/calculate_ab_rotation_02.py`
*   **Function:** Calculates `a` and `b` rotation coefficients for a specific `target_order2_norm_rotate` from raw Zernike moments. It uses the `eigen_root` function to solve polynomial equations.
*   **Numerical Flow (JAX Context):**
    1.  Determines `abconj_coef` based on `target_order2_norm_rotate` (even or odd) and calls `eigen_root` to find `abconj_sol`. This involves array indexing, conditional logic, complex conjugation, and polynomial root finding.
    2.  Calculates various powers and products of real and imaginary parts of `abconj_sol`. These are element-wise numerical operations.
    3.  Constructs `coef4` and `coef3` using complex arithmetic and powers.
    4.  Forms `bimbre_coef` array.
    5.  Calls `eigen_root` for each row of `bimbre_coef` to find `bimbre_sol_real`. This is a loop over polynomial root finding.
    6.  Filters `bimbre_sol_real` based on absolute value.
    7.  **Loop over `bimbre_sol_real`:**
        *   Calculates `bre` and `bim` using `np.sqrt`, division, and `np.power`.
        *   Constructs complex `b` and `a` using `np.vectorize(complex)` and complex conjugation.
        *   Appends `a` and `b` to lists.
    8.  Concatenates `a_list` and `b_list` into `ab_list`.
    *   **JAX Conversion Notes:**
        *   The initial `abconj_coef` calculation and `eigen_root` call are JAX-compatible.
        *   All element-wise numerical operations and power calculations are JAX-compatible.
        *   The loop for `bimbre_sol_real` and the subsequent loop for `a_list`/`b_list` would need to be vectorized using `jax.vmap` or `jax.lax.scan`.
        *   `np.vectorize(complex)` can be replaced by direct JAX complex number construction.
        *   `np.concatenate` is available in `jax.numpy`.
        *   This function is numerically intensive and would greatly benefit from JAX transformation, but requires careful handling of loops and conditional logic.

### `ZMPY3D_JAX/lib/__init__.py`
*   **Function:** This file serves as the package initializer for the `lib` directory. It imports and re-exports functions from individual Python files within `lib`, often renaming them for a cleaner API. It also exports CLI functions from the parent `ZMPY3D_JAX` directory.
*   **Numerical Flow (JAX Context):** This file is purely for module organization and does not contain any numerical computations. It would remain unchanged in a JAX conversion.
