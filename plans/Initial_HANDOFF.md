# ZMPY3D_JAX Fix Handoff

This document captures the current implementation gaps in `ZMPY3D_JAX` and the recommended sequence for bringing the project to a reliable, validated state.

## Current State

`ZMPY3D_JAX` contains the core pieces of the original ZMPY3D 3D Zernike pipeline:

- PDB CA-trace parsing.
- Residue Gaussian density voxelization.
- Bounding-box/geometric moment computation.
- Bounding-box moment to Zernike moment conversion.
- 3DZD invariant descriptor generation.
- Canterakis-style rotation normalization and superposition workflows.
- CLI wrappers for descriptors, shape scoring, and superposition.

The implementation is not yet production-ready. The main issues are a broken `calculate_ab_rotation_all` path, incomplete JAX conversion in rotation normalization, unreliable dtype configuration, biology-specific input assumptions, weak mathematical validation, and network-dependent integration tests.

## Priority 1: Fix `calculate_ab_rotation_all`

### Problem

`ZMPY3D_JAX/lib/calculate_ab_rotation_02_all.py` vmaps `ind_real` into `compute_ab_candidates_jax`, but `compute_ab_candidates_jax` declares `ind_real` as a static JIT argument:

- `ZMPY3D_JAX/lib/calculate_ab_rotation_02_all.py`
- `ZMPY3D_JAX/lib/calculate_ab_candidates_jax.py`

JAX cannot hash a `VmapTracer`, so tests for `calculate_ab_rotation_all` fail with:

```text
ValueError: Non-hashable static arguments are not supported
```

### Recommended Fix

Split the implementation into two paths:

1. A single-order JIT function where `ind_real` is static and called from Python loops.
2. A batched/all-orders function where `ind_real` is a traced array value and is not static.

The quickest safe remediation is to remove `static_argnames=("ind_real",)` from the function used by `vmap`, or create a second helper such as `compute_ab_candidates_batched_jax` for the all-orders path.

### Acceptance Criteria

- `pytest ZMPY3D_JAX/tests/module/test_calculate_ab_rotation.py` passes.
- Benchmark duplicate tests for AB rotation pass or are de-duplicated.
- `calculate_ab_rotation(zm, order)` and the matching entry from `calculate_ab_rotation_all(zm, order)` agree within explicit tolerance.
- No `VmapTracer` static-argument errors remain.

## Priority 2: Complete Rotation Normalization in JAX

### Problem

`ZMPY3D_JAX/lib/calculate_zm_by_ab_rotation01.py` still uses NumPy, Python loops, and `np.add.at`.

This function is central to Canterakis normalization and superposition. It may preserve original behavior in some paths, but it is not a complete JAX implementation and cannot be reliably JIT-compiled or accelerated.

### Recommended Fix

Port the function in stages:

1. Convert inputs with `jnp.asarray` using the project dtype config.
2. Replace conditional construction of `f_exp` with `jnp.where`.
3. Replace the rotation loop with `jax.vmap` over `ab_list`.
4. Replace `np.add.at(z_nlm, s_id, ...)` with `z_nlm.at[s_id].add(...)`.
5. Return a stacked JAX array or preserve the public list return with a wrapper around an internal JAX function.

Be careful around the current log-domain calculation:

- `a / b` and `a / conj(b)` produce divide-by-zero warnings for identity-like rotations.
- `log(0)` occurs when source moments are zero.
- Decide whether to preserve upstream behavior exactly or introduce numerically safer branches.

### Acceptance Criteria

- Existing `test_calculate_zm_by_ab_rotation.py` tests pass.
- Add an identity-rotation test that checks mathematically expected preservation without broad shape-permutation fallback.
- Add parity tests against `externals/ZMPY3D` for known fixtures.
- No warnings are emitted for valid identity/small-rotation cases unless explicitly expected and tested.

## Priority 3: Fix Runtime Dtype Configuration

### Problem

`ZMPY3D_JAX/config.py` mutates module globals:

```python
FLOAT_DTYPE = jnp.float32
COMPLEX_DTYPE = jnp.complex64
```

Some modules import these values directly:

```python
from ZMPY3D_JAX.config import FLOAT_DTYPE
from ZMPY3D_JAX.config import COMPLEX_DTYPE
```

After direct import, later calls to `configure_for_scientific_computing(enable_x64=True)` do not reliably update those modules.

### Recommended Fix

Use one consistent pattern across the package:

```python
import ZMPY3D_JAX.config as _config
```

Then reference:

```python
_config.FLOAT_DTYPE
_config.COMPLEX_DTYPE
```

Replace direct imports in at least:

- `ZMPY3D_JAX/lib/calculate_bbox_moment06.py`
- `ZMPY3D_JAX/lib/calculate_bbox_moment_2_zm05.py`
- `ZMPY3D_JAX/lib/calculate_molecular_radius03.py`
- `ZMPY3D_JAX/lib/get_ca_distance_info.py`
- `ZMPY3D_JAX/lib/get_transform_matrix_from_ab_list02.py`
- `ZMPY3D_JAX/lib/get_mean_invariant03.py`
- `ZMPY3D_JAX/lib/get_bbox_moment_xyz_sample01.py`

### Acceptance Criteria

- A test configures x64 after import and verifies new arrays are `float64`/`complex128`.
- A test configures x64 false and verifies new arrays are `float32`/`complex64`.
- No direct `from ZMPY3D_JAX.config import FLOAT_DTYPE` or `COMPLEX_DTYPE` imports remain outside compatibility shims.

## Priority 4: Remove Network Dependency From Tests

### Problem

`ZMPY3D_JAX/tests/conftest.py` downloads `6NT5.pdb` and `6NT6.pdb` from GitHub at test time, even though these files exist in the repository root.

This makes tests fail offline and in restricted environments.

### Recommended Fix

Change the fixture to prefer local files:

```python
repo_root = Path(__file__).resolve().parents[2]
local_path = repo_root / f"{name}.pdb"
```

Only download as an explicit fallback when local fixtures are missing, or remove download behavior entirely.

### Acceptance Criteria

- Integration tests do not require network access.
- `pytest -q` no longer errors because of `urllib.request.urlretrieve`.
- CI can run in offline mode for fixture-backed integration tests.

## Priority 5: Add Golden-Value and Mathematical Validation

### Problem

Many tests assert shapes, dtypes, non-null values, or broad deterministic behavior. They do not prove numerical correctness.

### Recommended Fix

Use the upstream submodule in `externals/ZMPY3D` as a reference implementation.

Add fixtures for:

- `6NT5.pdb`
- `6NT6.pdb`
- A tiny synthetic CA trace.
- A simple voxel object with known symmetries.
- An analytically simple density/shape where low-order moments can be checked.

Recommended test categories:

- JAX vs upstream NumPy parity for voxelization.
- JAX vs upstream NumPy parity for bbox moments.
- JAX vs upstream NumPy parity for Zernike raw/scaled moments.
- Rotation invariance under known rigid transforms.
- Descriptor stability under translation and rotation.
- Shape-score/superposition regression values.

### Acceptance Criteria

- Golden fixtures are committed or generated deterministically.
- Tolerances are explicit and documented.
- Tests fail on meaningful numerical drift, not only on shape mismatches.

## Priority 6: Clarify General vs Structural-Biology Scope

### Problem

The current input path is structural-biology-specific:

- `get_pdb_xyz_ca02.py` only reads `ATOM` lines and only CA atoms.
- Unknown residue names silently fall back to `ASP` in `fill_voxel_by_weight_density04.py`.

This is acceptable for a protein CA-trace descriptor workflow, but not for a general Zernike moments package.

### Recommended Fix

Define separate APIs:

1. Structural biology API:
   - PDB/mmCIF parsing.
   - CA-only or all-atom modes.
   - Explicit unknown-residue handling.
   - Ligand/nucleic-acid policy.

2. General shape API:
   - Accept a precomputed 3D voxel density array.
   - Accept point clouds plus weights/radii.
   - Avoid residue-specific assumptions.

Change unknown residue handling from silent fallback to one of:

- raise an explicit error,
- skip with warning,
- accept a caller-provided fallback residue,
- accept caller-provided radius/weight maps.

### Acceptance Criteria

- Public functions clearly state accepted input types.
- General voxel input can compute moments without PDB/residue machinery.
- Unknown residue behavior is explicit and tested.

## Suggested Work Order

1. Fix dtype config imports.
2. Fix offline integration fixtures.
3. Fix `calculate_ab_rotation_all`.
4. Add upstream parity tests for the functions touched above.
5. Port `calculate_zm_by_ab_rotation01.py` to JAX.
6. Add descriptor/superposition golden tests.
7. Add general voxel/point-cloud APIs if general-use support is required.

## Validation Commands

Run after each priority fix:

```bash
uv sync --extra dev --extra notebooks
uv run pytest ZMPY3D_JAX/tests/module/test_calculate_ab_rotation.py
uv run pytest ZMPY3D_JAX/tests/module/test_calculate_zm_by_ab_rotation.py
uv run pytest ZMPY3D_JAX/tests/integration/test_superposition.py
uv run pytest -q
```

Use `externals/ZMPY3D` for parity checks:

```bash
git submodule update --init --recursive
```

## Definition of Done

The codebase can be called a complete implementation only when:

- The full test suite passes offline.
- Core numerical outputs match the upstream NumPy implementation within documented tolerances.
- Rotation normalization and all-order AB candidate generation are correct and stable.
- Dtype configuration works predictably.
- Public documentation clearly distinguishes structural-biology workflows from general shape workflows.
- General-purpose entry points exist if the project claims general Zernike moment support.
