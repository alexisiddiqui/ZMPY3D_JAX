# ZMPY3D_JAX Stabilization Handoff

This document records the verified state of the JAX port and the remaining validation work. CPU
is the default JAX platform; callers should enable x64 before numerical work when scientific
precision is required.

## Completed Stabilization

- `calculate_ab_rotation_all` uses a traced internal candidate function under `vmap`; the earlier
  non-hashable static `ind_real` diagnosis no longer applies.
- AB candidate filtering uses a precision-aware tolerance so float32 companion-matrix solves do
  not retain nominally-zero duplicate roots.
- Rotation of Zernike moments is evaluated by a batched JAX kernel while the public API retains its
  list-of-NumPy-arrays return contract.
- Exact identity rotations preserve the raw moments in the documented `(m, l, n)` output layout.
- Runtime dtype changes propagate through library modules and the package-level `FLOAT_DTYPE` and
  `COMPLEX_DTYPE` attributes.
- Integration fixtures use committed PDB files, generated artifacts use pytest temporary
  directories, and benchmarks are excluded from the default correctness suite.
- Unknown residue names raise `ValueError` instead of silently receiving ASP properties.
- Core voxel, bbox-moment, Zernike-conversion, descriptor, AB-candidate, and rotation stages have
  direct parity coverage against `externals/ZMPY3D`.

## Supported Scope

The supported input workflow is protein CA traces in legacy PDB text format. The parser accepts
blank or `A` alternate locations and ignores secondary alternate locations. General point-cloud,
arbitrary-volume, mmCIF, nucleic-acid, ligand, and all-atom APIs are outside the current scope.

## Validation

Install the locked environment and run the offline correctness suite:

```bash
uv sync --locked --extra dev --extra notebooks
uv run --no-sync pytest -q
```

Run performance tests separately:

```bash
uv run --no-sync pytest -m benchmark
```

Focused numerical checks:

```bash
uv run --no-sync pytest ZMPY3D_JAX/tests/module/test_upstream_parity.py
uv run --no-sync pytest ZMPY3D_JAX/tests/module/test_calculate_ab_rotation.py
uv run --no-sync pytest ZMPY3D_JAX/tests/module/test_calculate_zm_by_ab_rotation.py
```

## Remaining Work

- Add CI jobs for the offline correctness suite on supported Python versions.
- Record performance baselines by hardware class rather than treating timing as correctness.
- Add order-20 parity coverage where CI memory and runtime budgets permit.
- Add a separate API design before expanding beyond protein CA-trace inputs.

## Definition of Done

- The default suite passes without network access, numerical warnings, or source-tree artifacts.
- JAX outputs match the upstream NumPy implementation within explicit per-stage tolerances.
- x32 and x64 configurations behave consistently while CPU remains the default platform.
- Rotation normalization is finite for valid candidates and exact for identity rotation.
