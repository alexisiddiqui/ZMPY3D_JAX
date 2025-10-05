# ZMPY3D_JAX Library Components

## Overview
This directory contains the core computational library for ZMPY3D_JAX. Files are categorized by their primary function: numerical operations, utilities, or I/O operations.

## JAX Conversion Progress Tracker

### Legend
- ✅ **Converted**: Fully converted to JAX
- 🔄 **In Progress**: Currently being converted
- ⏳ **Pending**: Not yet started
- ❌ **Not Applicable**: No conversion needed (pure Python/I/O)

---

## File Categories

### 🔢 Core Numerical Operations (High Priority for JAX Conversion)

These files contain the main mathematical computations and are critical for JAX conversion to achieve performance gains.

#### ✅ `calculate_ab_candidates_jax.py`
**Purpose**: Computes ALL candidate a/b values with validity mask.
**Conversion Priority**: ⭐⭐⭐⭐⭐ (Critical - hot path)
**JAX Conversion Notes**:
- New file, fully implemented in JAX.
- Used by `calculate_ab_rotation_02.py` and `calculate_ab_rotation_02_all.py`.

---

#### ✅ `calculate_bbox_moment06.py`
**Purpose**: Calculates 3D bounding box moments using tensor operations  
**Conversion Priority**: ⭐⭐⭐⭐⭐ (Critical - hot path)  
**NumPy Operations**:
- `np.zeros()` - Array initialization
- `np.diff()` - Differencing along axes (3x)
- `np.arange()` - Index generation
- `np.power()` - Element-wise exponentiation
- `np.tensordot()` - Tensor contraction (nested, 3 levels) ⚡ **Performance critical**
- `np.meshgrid()` - Coordinate grid generation
- Array slicing and transposition

**JAX Conversion Notes**:
- Fully converted to JAX.
- Uses `jnp.zeros`, `jnp.diff`, `jnp.power`, `jnp.arange`, `jnp.tensordot`, `jnp.meshgrid`, `jnp.transpose`.
- Immutable array updates handled with `.at[]` syntax.
- `np.tensordot` → `jnp.tensordot` (major speedup expected)
- Consider `jax.jit` for entire function

---

#### 🔄 `calculate_bbox_moment_2_zm05.py`
**Purpose**: Converts bounding box moments to Zernike moments  
**Conversion Priority**: ⭐⭐⭐⭐⭐ (Critical - hot path)  
**NumPy Operations**:
- `np.zeros()` - Array initialization
- `np.reshape()` - Array reshaping
- `np.add.at()` - In-place accumulation ⚠️ **JAX immutability issue**
- Complex array operations
- Element-wise multiplication
- Conditional assignment (NaN handling)

**JAX Conversion Notes**:
- `np.add.at()` → `jax.ops.index_add()` (critical change)
- Use `jnp.where()` for conditional NaN assignment
- All complex operations supported in JAX

**JAX Conversion Challenges**:
- ⚠️ In-place updates (`np.add.at`) require functional refactoring

**Code Example (NumPy)**:
```python
# In-place accumulation in calculate_bbox_moment_2_zm05.py
np.add.at(zm_geo_sum, g_cache_complex_index - 1, zm_geo)
```
This `np.add.at` operation performs an in-place addition, which is not directly supported in JAX due to its immutable array design. It needs to be replaced with a functional equivalent like `jax.ops.index_add`.

---

#### 🔄 `calculate_zm_by_ab_rotation01.py`
**Purpose**: Calculates rotated Zernike moments from rotation coefficients  
**Conversion Priority**: ⭐⭐⭐⭐⭐ (Critical - hot path)  
**NumPy Operations**:
- `np.log()` - Logarithmic transformation
- `np.exp()` - Exponential (in loop)
- `np.add.at()` - In-place accumulation ⚠️ **JAX immutability issue**
- Complex arithmetic (conjugate, real, multiplication)
- Element-wise operations
- Array indexing and slicing

**JAX Conversion Notes**:
- Loop over rotations → `jax.lax.scan()` or `jax.vmap()`
- `np.add.at()` → `jax.ops.index_add()` or functional equivalent
- Consider `jax.scipy.special.logsumexp()` for numerical stability
- Use `jnp.where()` for conditional logic

**JAX Conversion Challenges**:
- ⚠️ Loop vectorization required
- ⚠️ In-place accumulation needs functional approach

**Code Example (NumPy)**:
```python
# Inside the loop over rotations in calculate_zm_by_ab_rotation01.py
# This performs an in-place update within a loop
np.add.at(z_nlm, s_id, np.exp(nlm))
```
This example shows both a loop that needs vectorization (e.g., with `jax.lax.scan` or `jax.vmap`) and an in-place accumulation (`np.add.at`) that needs to be refactored into a functional update using `jax.ops.index_add` or a similar approach.

---

#### ✅ `calculate_ab_rotation_02.py`
**Purpose**: Computes rotation coefficients (a, b) for a specific order  
**Conversion Priority**: ⭐⭐⭐⭐ (High priority)  
**NumPy Operations**:
- `np.abs()` - Absolute value
- `np.sqrt()` - Square root
- `np.power()` - Exponentiation
- `np.vectorize(complex)` - Complex number construction ⚠️ **Not JAX compatible**
- `np.concatenate()` - Array concatenation
- Complex conjugation
- Polynomial root finding via `eigen_root()`

**JAX Conversion Notes**:
- Uses `compute_ab_candidates_jax` for core logic, which is JAX-compatible.
- `np.vectorize(complex)` → Direct JAX complex construction
- Loop over solutions → `jax.lax.scan()` or `jax.vmap()`
- Filter operations → `jnp.where()` with masking

**JAX Conversion Challenges**:
- ⚠️ Multiple nested loops need vectorization
- ⚠️ Conditional filtering requires functional approach

**Code Example (NumPy)**:
```python
# Simplified loop and conditional filtering in calculate_ab_rotation_02.py
# This pattern needs to be vectorized in JAX.
for sol_real in bimbre_sol_real:
    if np.abs(sol_real) > 1e-6:
        # ... calculations and appending to lists ...
```
This illustrates the need to vectorize loops (e.g., with `jax.lax.scan` or `jax.vmap`) and handle conditional filtering using `jax.numpy.where` or similar functional constructs in JAX.

---

#### ✅ `calculate_ab_rotation_02_all.py`
**Purpose**: Computes rotation coefficients for all orders  
**Conversion Priority**: ⭐⭐⭐⭐ (High priority)  
**NumPy Operations**:
- Same as `calculate_ab_rotation_02.py`
- Additional outer loop over orders
- List concatenation

**JAX Conversion Notes**:
- Uses `compute_ab_candidates_jax` for core logic, which is JAX-compatible.
- Nested function → Ensure JAX compatibility
- Outer loop over orders → `jax.vmap()` if possible
- Share conversion strategy with `calculate_ab_rotation_02.py`

**JAX Conversion Challenges**:
- ⚠️ Double nested loops (orders + solutions)

**Code Example (NumPy)**:
```python
# Outer loop over orders and inner loops from calculate_ab_rotation_02.py
# This double-nested loop structure needs careful vectorization in JAX.
for ind_real in ind_real_all:
    ab_list.append(get_ab_list_by_ind_real(ind_real, ...))
```
This highlights the challenge of vectorizing both the outer loop over different orders and the inner loops present in `get_ab_list_by_ind_real` (which is called for each order). `jax.vmap` or `jax.lax.scan` would be crucial here.

---

#### ✅ `eigen_root.py`
**Purpose**: Polynomial root finding via eigenvalue decomposition  
**Conversion Priority**: ⭐⭐⭐⭐ (High priority)  
**NumPy Operations**:
- `np.reshape()` - Array reshaping
- `np.diag()` - Diagonal matrix creation
- `np.linalg.eigvals()` - Eigenvalue computation ⚡ **Performance critical**
- Array assignment

**JAX Conversion Notes**:
- Fully converted to JAX.
- Uses `jnp.asarray`, `jnp.diag`, `jnp.linalg.eigvals`, `jax.lax.cond`, and `jax.vmap`.
- `np.diag()` → `jnp.diag()`
- `np.linalg.eigvals()` → `jnp.linalg.eigvals()` (GPU acceleration available)
- Direct conversion, minimal changes needed

---

#### ✅ `calculate_molecular_radius03.py`
**Purpose**: Computes average and maximum molecular radii  
**Conversion Priority**: ⭐⭐⭐ (Medium priority)  
**NumPy Operations**:
- `np.where()` - Boolean indexing
- `np.stack()` - Array stacking
- `np.sum()` - Reduction operation
- `np.sqrt()` - Square root
- `np.max()` - Maximum value
- Element-wise power (`**2`)

**JAX Conversion Notes**:
- Fully converted to JAX.
- Uses `jnp.asarray`, `jnp.where`, `jnp.stack`, `jnp.sum`, `jnp.sqrt`, `jnp.max`, and `jnp.std`.
- Direct `jax.numpy` conversion
- Boolean indexing well supported
- All operations efficiently implemented in JAX

---

#### ✅ `get_mean_invariant03.py`
**Purpose**: Calculates mean and std of Zernike moment arrays  
**Conversion Priority**: ⭐⭐⭐ (Medium priority)  
**NumPy Operations**:
- `np.stack()` - Array stacking
- `np.abs()` - Absolute value
- `np.mean()` - Mean along axis
- `np.std()` - Standard deviation (ddof=1)

**JAX Conversion Notes**:
- Fully converted to JAX.
- Uses `jnp.stack`, `jnp.abs`, `jnp.mean`, and `jnp.std`.
- Direct `jax.numpy` conversion
- `ddof` parameter supported in `jnp.std()`
- Consider `jax.jit` for performance

---

#### ✅ `get_ca_distance_info.py`
**Purpose**: Calculates geometric descriptors from CA coordinates  
**Conversion Priority**: ⭐⭐⭐ (Medium priority)  
**NumPy Operations**:
- `np.mean()` - Center of mass
- `np.sqrt()`, `np.sum()`, `**2` - Distance calculation
- `np.percentile()` - Percentile computation
- `np.std()` - Standard deviation
- Statistical moments (skewness, kurtosis)

**JAX Conversion Notes**:
- Fully converted to JAX.
- Uses `jnp.mean`, `jnp.sqrt`, `jnp.sum`, `jnp.percentile`, `jnp.std`, and `jnp.where`.
- Most operations have direct JAX equivalents
- `np.percentile()` → `jnp.percentile()`
- Consider `jax.jit` for entire function

---

#### 🔄 `get_3dzd_121_descriptor02.py`
**Purpose**: Calculates 3DZD 121 rotation-invariant descriptor  
**Conversion Priority**: ⭐⭐⭐ (Medium priority)  
**NumPy Operations**:
- `np.nan_to_num()` or manual NaN replacement
- `np.abs()` - Absolute value
- `**2` - Element-wise power
- `np.sum()` - Reduction (axis-specific)
- `np.sqrt()` - Square root
- Array slicing and conditional assignment

**JAX Conversion Notes**:
- `np.nan_to_num()` → `jnp.nan_to_num()` or `jnp.where(jnp.isnan(x), 0, x)`
- Conditional assignment → `jnp.where()`
- Array slicing well supported

---

### 🔧 Utility Functions (Mixed Priority)

#### 🔄 `fill_voxel_by_weight_density04.py`
**Purpose**: Fills 3D voxel grid with density values  
**Conversion Priority**: ⭐⭐⭐⭐ (High - hot path but challenging)  
**NumPy Operations**:
- `np.min()`, `np.max()` - Bounding box calculation
- `np.ceil()` - Ceiling operation
- `np.zeros()` - Array initialization
- `np.fix()`, `np.round()` - Rounding operations
- Array slicing and in-place addition (`+=`) ⚠️ **JAX immutability issue**
- Dictionary lookups in loop

**JAX Conversion Notes**:
- Loop over atoms → `jax.lax.scan()` (complex refactoring)
- In-place `+=` → `jax.lax.dynamic_update_slice()`
- Dictionary lookups → Convert to array indexing
- Pre-compute residue boxes as arrays

**JAX Conversion Challenges**:
- ⚠️ Major refactoring needed for atom loop
- ⚠️ Dynamic slicing required for voxel updates
- ⚠️ Dictionary to array conversion needed

**Code Example (NumPy)**:
```python
# Inside the atom loop in fill_voxel_by_weight_density04.py
# This performs an in-place update on the voxel grid
voxel3d[start[0]:end[0], start[1]:end[1], start[2]:end[2]] += aa_box * weight_multiplier
```
This in-place array modification is a common pattern in NumPy but is problematic in JAX due to its functional programming paradigm and immutable arrays. JAX requires explicit functional updates, often using `jax.lax.dynamic_update_slice` or `jax.ops.index_add` for such operations, which can significantly change the code structure within loops.

---

#### 🔄 `calculate_box_by_grid_width.py`
**Purpose**: Generates Gaussian density boxes for residues  
**Conversion Priority**: ⭐⭐⭐ (Medium priority)  
**NumPy Operations**:
- `np.sqrt()`, `**2` - Distance calculations
- `np.ceil()` - Rounding
- `np.mgrid[]` - Grid generation
- `np.exp()` - Gaussian calculation
- `np.zeros()` - Array initialization
- Boolean indexing and conditional assignment

**JAX Conversion Notes**:
- Loop over residues → `jax.vmap()` (if uniform box sizes)
- `np.mgrid` → `jnp.mgrid` (supported)
- Boolean indexing → `jnp.where()`
- Consider pre-computation and caching

**JAX Conversion Challenges**:
- ⚠️ Variable box sizes may complicate vectorization

**Code Example (NumPy)**:
```python
# Loop over residue names in calculate_box_by_grid_width.py
# Box sizes can vary per residue, complicating direct vmap.
for residue_name in residue_name_list:
    # ... calculations for residue_unit_box ...
    residue_boxes[residue_name] = residue_unit_box
```
This loop iterates through residues, and the `residue_unit_box` can have different dimensions depending on the residue's properties. This variability in shape makes direct `jax.vmap` challenging and might require `jax.lax.scan` or padding to a common maximum size.

---

#### 🔄 `get_residue_gaussian_density_cache02.py`
**Purpose**: Pre-calculates Gaussian density caches  
**Conversion Priority**: ⭐⭐ (Lower - preprocessing step)  
**NumPy Operations**:
- `np.sum()` - Reduction
- Element-wise multiplication and division
- `**3` - Exponentiation
- Calls `calculate_box_by_grid_width()` in loop

**JAX Conversion Notes**:
- Can remain mostly NumPy (preprocessing)
- Or convert loop → `jax.lax.scan()` with `calculate_box_by_grid_width` JAX-ified
- Output arrays convert to JAX arrays

---

#### ✅ `get_bbox_moment_xyz_sample01.py`
**Purpose**: Generates normalized sample coordinates  
**Conversion Priority**: ⭐⭐ (Lower priority)  
**NumPy Operations**:
- `np.arange()` - Array generation
- Element-wise subtraction and division

**JAX Conversion Notes**:
- Fully converted to JAX.
- Uses `jnp.arange`, subtraction, and division.
- Trivial conversion to `jax.numpy`
- Direct replacement

---

#### ✅ `get_transform_matrix_from_ab_list02.py`
**Purpose**: Constructs 4x4 transformation matrix from a, b coefficients  
**Conversion Priority**: ⭐⭐⭐ (Medium priority)  
**NumPy Operations**:
- Complex arithmetic (real, imag, conjugate)
- `np.reshape()` - Matrix construction
- `np.linalg.inv()` - Matrix inversion ⚡ **Performance critical**
- Element-wise operations

**JAX Conversion Notes**:
- Fully converted to JAX.
- Uses `jnp.array`, `jnp.real`, `jnp.imag`, `jnp.reshape`, `jnp.zeros`, `.at[]`, and `jnp.linalg.inv`.
- Direct conversion to `jax.numpy`
- `np.linalg.inv()` → `jnp.linalg.inv()` (GPU acceleration)
- Complex operations well supported

---

#### ❌ `get_total_residue_weight.py`
**Purpose**: Calculates total residue weight  
**Conversion Priority**: ⭐ (Low - not applicable)  
**Notes**: Returns a single float (aggregate total). No JAX conversion required.

---

#### ⏳ `get_descriptor_property.py`
**Purpose**: Returns descriptor weights and indices  
**Conversion**: Not applicable (static configuration)  
**NumPy Operations**:
- `np.array()` - Static array definition

**JAX Notes**: Convert to `jnp.array()` at initialization

---

### 📁 I/O Operations (No JAX Conversion Needed)

#### ❌ `get_pdb_xyz_ca02.py`
**Purpose**: Parses PDB files for CA coordinates  
**Conversion**: Not applicable (I/O and string processing)  
**NumPy Operations**:
- `np.array()` - Final conversion to array

**JAX Notes**: Output can be converted to JAX array after parsing

---

#### ❌ `set_pdb_xyz_rot_m_01.py`
**Purpose**: Applies rotation matrix and writes PDB  
**Conversion**: Partial (matrix multiplication only)  
**NumPy Operations**:
- `np.hstack()` - Homogeneous coordinates
- `@` operator - Matrix multiplication ⚡ **Could be JAX**
- `np.round()` - Rounding

**JAX Notes**:
- Keep I/O in NumPy
- Matrix multiplication could use JAX if rotation matrix is JAX array
- Or convert result back to NumPy for I/O

---

#### ❌ `read_file_as_string_list.py`
**Purpose**: File reading utility  
**Conversion**: Not applicable (pure I/O)

---

#### ❌ `write_string_list_as_file.py`
**Purpose**: File writing utility  
**Conversion**: Not applicable (pure I/O)

---

#### ❌ `__init__.py`
**Purpose**: Package initialization  
**Conversion**: Not applicable (imports only)

---

## Conversion Strategy

### Phase 1: Core Numerical Operations (Weeks 1-3)
1. ✅ `eigen_root.py` - Low complexity, high impact
2. ✅ `get_bbox_moment_xyz_sample01.py` - Simple conversion
3. ✅ `calculate_molecular_radius03.py` - Moderate complexity
4. ✅ `get_mean_invariant03.py` - Simple reduction operations
5. ✅ `get_ca_distance_info.py` - Statistical operations
6. ✅ `calculate_ab_candidates_jax.py` - New JAX implementation

### Phase 2: Zernike Moments (Weeks 4-6)
7. ✅ `calculate_bbox_moment06.py` - Critical tensordot operations
8. 🔄 `calculate_bbox_moment_2_zm05.py` - Handle `np.add.at` challenge
9. 🔄 `get_3dzd_121_descriptor02.py` - Descriptor calculation

### Phase 3: Rotation Calculations (Weeks 7-9)
10. ✅ `calculate_ab_rotation_02.py` - Complex loop vectorization
11. ✅ `calculate_ab_rotation_02_all.py` - Extend to all orders
12. 🔄 `calculate_zm_by_ab_rotation01.py` - Critical rotation application

### Phase 4: Voxelization (Weeks 10-12)
13. 🔄 `calculate_box_by_grid_width.py` - Gaussian box generation
14. 🔄 `fill_voxel_by_weight_density04.py` - Most challenging conversion
15. 🔄 `get_residue_gaussian_density_cache02.py` - Cache generation

### Phase 5: Utilities & Optimization (Weeks 13-14)
1. ✅ `get_transform_matrix_from_ab_list02.py` - Matrix operations
2. ❌ `get_total_residue_weight.py` - Returns float; no conversion needed
3. ⏳ `get_descriptor_property.py` - Convert static data dictionaries to arrays
4. ✅ End-to-end testing and optimization

---

## Key JAX Conversion Patterns

### Pattern 1: In-place Updates
```python
# NumPy (in-place)
np.add.at(array, indices, values)

# JAX (functional)
array = jax.ops.index_add(array, indices, values)
```

### Pattern 2: Loop Vectorization
```python
# NumPy (loop)
for i in range(n):
    result[i] = compute(data[i])

# JAX (vmap)
result = jax.vmap(compute)(data)
```

### Pattern 3: Conditional Assignment
```python
# NumPy (boolean indexing)
array[condition] = value

# JAX (where)
array = jnp.where(condition, value, array)
```

### Pattern 4: Dictionary to Array Lookup
```python
# NumPy (dict)
weights = [weight_map[name] for name in names]

# JAX (array indexing)
# Pre-process: name → index mapping
weights = weight_array[name_indices]
```

---

## Performance Expectations

### Expected Speedups (GPU)
- **Tensor operations** (`tensordot`): 5-20x
- **Linear algebra** (`eigvals`, `inv`): 3-10x  
- **Element-wise ops**: 2-5x
- **Vectorized loops**: 10-50x

### Memory Considerations
- JAX uses device memory (GPU/TPU)
- May need batching for large structures
- Watch for memory transfers (host ↔ device)

---

## Testing Strategy

### Unit Tests per File
- [ ] Create test cases with known inputs/outputs
- [ ] Compare NumPy vs JAX outputs (tolerance: 1e-6)
- [ ] Test edge cases (NaN, zero, complex)

### Integration Tests
- [ ] Full pipeline: PDB → Voxel → Moments → Descriptor
- [ ] Shape score calculation
- [ ] Superimposition accuracy

### Performance Benchmarks
- [ ] Time each converted function
- [ ] Profile memory usage
- [ ] Compare CPU vs GPU performance

---

## Notes & Gotchas

### JAX Limitations
- ⚠️ No in-place updates (immutable arrays)
- ⚠️ Dynamic shapes difficult (use static or `vmap`)
- ⚠️ Control flow needs special handling (`lax.cond`, `lax.scan`)
- ⚠️ RNG different from NumPy (use `jax.random`)

### Numerical Precision
- JAX defaults to float32 (NumPy: float64)
- Set `jax.config.update("jax_enable_x64", True)` for float64
- Complex128 available but check compatibility

### Debugging
- Use `jax.debug.print()` inside JIT
- `.block_until_ready()` for accurate timing
- `jax.disable_jit()` for debugging

---

## Progress Tracking

| File | Status | Conversion Date | Notes | Performance Gain |
|------|--------|----------------|-------|------------------|
| `calculate_ab_candidates_jax.py` | ✅ | 2025-10-05 | New JAX implementation | - |
| `eigen_root.py` | ✅ | 2025-10-05 | Fully converted to JAX | High |
| `calculate_bbox_moment06.py` | ✅ | 2025-10-05 | Fully converted to JAX | High |
| `calculate_bbox_moment_2_zm05.py` | 🔄 | - | `index_add` challenge remains | - |
| `calculate_zm_by_ab_rotation01.py` | 🔄 | - | Loop vectorization and `index_add` challenge remains | - |
| `calculate_ab_rotation_02.py` | ✅ | 2025-10-05 | Uses `compute_ab_candidates_jax` | High |
| `calculate_ab_rotation_02_all.py` | ✅ | 2025-10-05 | Uses `compute_ab_candidates_jax` | High |
| `calculate_molecular_radius03.py` | ✅ | 2025-10-05 | Fully converted to JAX | Medium |
| `get_mean_invariant03.py` | ✅ | 2025-10-05 | Fully converted to JAX | Medium |
| `get_ca_distance_info.py` | ✅ | 2025-10-05 | Fully converted to JAX | Medium |
| `get_3dzd_121_descriptor02.py` | 🔄 | - | Conditional assignment needs `jnp.where` | - |
| `fill_voxel_by_weight_density04.py` | 🔄 | - | Major refactoring needed for atom loop and dynamic slicing | - |
| `calculate_box_by_grid_width.py` | 🔄 | - | Loop vectorization and variable box sizes challenge | - |
| `get_bbox_moment_xyz_sample01.py` | ✅ | 2025-10-05 | Fully converted to JAX | Low |
| `get_transform_matrix_from_ab_list02.py` | ✅ | 2025-10-05 | Fully converted to JAX | Medium |
| `get_residue_gaussian_density_cache02.py` | 🔄 | - | Depends on `calculate_box_by_grid_width` conversion | - |
| `get_total_residue_weight.py` | ❌ | - | Returns a float; no JAX conversion required | - |
| `get_descriptor_property.py` | ⏳ | - | Convert static NumPy arrays to JAX arrays | - |
| `get_residue_weight_map01.py` | ❌ | - | Static data, convert to JAX array for use in JAX functions | - |
| `get_residue_radius_map01.py` | ❌ | - | Static data, convert to JAX array for use in JAX functions | - |
| `get_global_parameter02.py` | ❌ | - | Configuration, use `jnp.pi` for constant calculation | - |
| `get_pdb_xyz_ca02.py` | ❌ | - | I/O and string processing, output converted to JAX array | - |
| `set_pdb_xyz_rot_m_01.py` | ❌ | - | I/O, matrix multiplication could use JAX | - |
| `read_file_as_string_list.py` | ❌ | - | Pure I/O | - |
| `write_string_list_as_file.py` | ❌ | - | Pure I/O | - |
| `__init__.py` | ❌ | - | Imports only | - |

---

## Resources

### JAX Documentation
- [JAX Quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- [JAX NumPy API](https://jax.readthedocs.io/en/latest/jax.numpy.html)
- [Common Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)

### Conversion Guides
- [NumPy to JAX Migration](https://jax.readthedocs.io/en/latest/jax-101/01-jax-basics.html)
- [Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)
- `jax.disable_jit()` for debugging

---

## Progress Tracking

| File | Status | Conversion Date | Notes | Performance Gain |
|------|--------|----------------|-------|------------------|
| `calculate_ab_candidates_jax.py` | ✅ | 2025-10-05 | New JAX implementation | - |
| `eigen_root.py` | ✅ | 2025-10-05 | Fully converted to JAX | High |
| `calculate_bbox_moment06.py` | ✅ | 2025-10-05 | Fully converted to JAX | High |
| `calculate_bbox_moment_2_zm05.py` | ✅ | - | `index_add` challenge remains | - |
| `calculate_zm_by_ab_rotation01.py` | 🔄 | - | Loop vectorization and `index_add` challenge remains | - |
| `calculate_ab_rotation_02.py` | 🔄 | 2025-10-05 | Uses `compute_ab_candidates_jax` but not vectorised yet |  |
| `calculate_ab_rotation_02_all.py` | 🔄 | 2025-10-05 | Uses `compute_ab_candidates_jax` but not vectorised yet |  |
| `calculate_molecular_radius03.py` | ✅ | 2025-10-05 | Fully converted to JAX | Medium |
| `get_mean_invariant03.py` | ✅ | 2025-10-05 | Fully converted to JAX | Medium |
| `get_ca_distance_info.py` | ✅ | 2025-10-05 | Fully converted to JAX | Medium |
| `get_3dzd_121_descriptor02.py` | ✅ | - | Converted | - |
| `fill_voxel_by_weight_density04.py` | 🔄 | - | used for featurising - low priority | - |
| `calculate_box_by_grid_width.py` | 🔄 | - | used for featurising - low priority. Loop vectorization and variable box sizes challenge | - |
| `get_bbox_moment_xyz_sample01.py` | ✅ | 2025-10-05 | Fully converted to JAX | Low |
| `get_transform_matrix_from_ab_list02.py` | ✅ | 2025-10-05 | Fully converted to JAX | Medium |
| `get_residue_gaussian_density_cache02.py` | 🔄 | - | Used for featurising - low priority. Depends on `calculate_box_by_grid_width` conversion.  | - |
| `get_total_residue_weight.py` | ⏳ | - |used for featurising - low priority| - |
| `get_descriptor_property.py` | ✅ | - | Converted| - |
| `get_residue_weight_map01.py` | ❌ | - | Static data, convert to JAX array for use in JAX functions | - |
| `get_residue_radius_map01.py` | ❌ | - | Static data, convert to JAX array for use in JAX functions | - |
| `get_global_parameter02.py` | ❌ | - | Configuration, use `jnp.pi` for constant calculation | - |
| `get_pdb_xyz_ca02.py` | ❌ | - | I/O and string processing, output converted to JAX array | - |
| `set_pdb_xyz_rot_m_01.py` | ❌ | - | I/O, matrix multiplication could use JAX | - |
| `read_file_as_string_list.py` | ❌ | - | Pure I/O | - |
| `write_string_list_as_file.py` | ❌ | - | Pure I/O | - |
| `__init__.py` | ❌ | - | Imports only | - |

---

## Resources

### JAX Documentation
- [JAX Quickstart](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- [JAX NumPy API](https://jax.readthedocs.io/en/latest/jax.numpy.html)
- [Common Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)

### Conversion Guides
- [NumPy to JAX Migration](https://jax.readthedocs.io/en/latest/jax-101/01-jax-basics.html)
- [Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)
