# ZMPY3D JAX Regression and Performance Handoff

Date: 2026-07-17

## Purpose

The upstream numerical regression suite is now broad enough to protect the supported CPU
workflow. The next phase should characterize and improve CPU performance without weakening the
validated numerical behavior.

## Verified State

The regression implementation is split into three parts:

- `ZMPY3D_JAX/tests/utils/upstream_regression.py` contains independent JAX and upstream pipeline
  runners plus deterministic inputs and shared cache loading.
- `ZMPY3D_JAX/tests/integration/test_upstream_regression.py` enforces numerical parity.
- `ZMPY3D_JAX/tests/benchmark/test_benchmark_regression_upstream.py` records informational CPU
  timing results.

Correctness coverage includes:

- Mixed-residue synthetic traces at grid widths `0.25`, `0.5`, and `1.0`.
- Translated and rotated synthetic coordinates.
- The committed `6NT5.pdb` and `6NT6.pdb` structures.
- Maximum orders 6 and 20; order 20 is marked `slow`.
- PDB parsing, residue caches, voxelization, bbox moments, radius, sphere samples, raw/scaled
  Zernike moments, 3DZD descriptors, normalization orders 2 through 5, AB candidate sets, and all
  canonicalized candidate rotations.
- Independent AB generation by JAX and upstream; the reference rotation is no longer given a
  JAX-derived AB pair.

CPU remains the default backend. Regression characterization explicitly enables x64 because it is
the supported scientific-precision configuration. Timing remains informational and has no
hardware-dependent failure threshold.

Latest verification:

```text
Default suite:             145 passed, 68 deselected
Order-20 regression tier:  2 passed
CPU timing benchmark:      3 passed
```

Latest measurement on a 12th Gen Intel Core i7-12700KF:

```text
Shared input/cache setup:   12.33 ms
JAX setup:                  77.16 ms
JAX rotation-cache setup:   10.97 ms
JAX bbox-to-ZM cache setup: 55.26 ms
Upstream setup:             69.64 ms
JAX first execution:         0.88 s
Upstream first execution:   18.06 ms
JAX warmed median:          10.93 ms
Upstream warmed median:     18.11 ms
JAX/upstream time ratio:     0.60
```

Latest structured result:
`upstream_regression_benchmark_20260717T122606_546149Z.json`.

Setup and first execution are measured independently. Warmed JAX is about 1.66 times faster than
warmed upstream NumPy for this single-protein CPU workload. Rotation and bbox-to-ZM cache
materialization are explicit one-time setup costs and are excluded from warmed stage timing.

## Completed Performance Diagnosis

The benchmark now emits schema-v3 JSON with separate shared setup, implementation setup, JAX
rotation-cache setup, first execution, warmed end-to-end latency, and synchronized stage profiles.
Ratio fields are explicit:
`jax_over_upstream_time_ratio` and `upstream_over_jax_time_ratio`.

The synchronized stage profile ranks bottlenecks by positive median excess time. It is diagnostic;
stage medians should not be summed or substituted for end-to-end latency.

Latest ranking:

| Rank | Stage | JAX median | Upstream median | Positive-gap contribution |
| ---: | --- | ---: | ---: | ---: |
| 1 | Descriptor | 0.805 ms | 0.031 ms | 96.23% |
| 2 | Bbox to ZM | 0.108 ms | 0.078 ms | 3.77% |
| 3 | ZM rotation | 0.406 ms | 0.447 ms | 0% |
| 4 | AB candidates | 0.107 ms | 0.297 ms | 0% |

AB candidate generation now measures 0.107 ms versus 0.297 ms upstream and has no positive
contribution to the remaining CPU gap.

### Completed AB root optimization

Internal profiling showed that the unjitted initial polynomial root solve consumed approximately
33.0 ms of the former 35.6 ms public AB call. The candidate kernel itself took about 0.021 ms and
dynamic filtering about 0.66 ms.

`eigen_root` now uses a dtype-aware, fixed-shape jitted companion-matrix kernel. Batched root
solving uses that kernel directly, so both `calculate_ab_rotation` and
`calculate_ab_rotation_all` benefit without changing their public return contracts.

Results:

- AB stage: 36.47 ms to 1.38 ms, a 96.2% reduction.
- Warm full pipeline: 88.94 ms to 57.34 ms, a 35.5% reduction.
- JAX/upstream full-pipeline ratio: 5.13 to 3.66.
- The agreed 5 ms stop target was satisfied, so whole-AB kernel fusion was not added.

### Completed rotation cache and batch optimization

The vectorized ZM rotation kernel is now jitted, and its binomial, CLM, and index caches are
materialized once as an immutable JAX cache. Internal descriptor and superposition workflows use a
fixed JAX rotation batch; the original list-of-NumPy-arrays function remains a compatibility
wrapper.

A 72-rotation 6NT5 microprofile measured 10.61 ms when the compiled kernel received NumPy cache
arguments on every call and 0.57 ms when it received prepared JAX arrays. In the full benchmark's
normalization-order-5 stage, warmed rotation now measures 0.312 ms versus 0.355 ms upstream.

Results:

- ZM rotation: 16.51 ms to 0.312 ms, a 98.1% reduction.
- Warm full pipeline: 57.34 ms to 46.58 ms, an 18.8% reduction.
- JAX/upstream full-pipeline ratio: 3.66 to 1.90.
- Rotation now ranks eighth with no positive contribution to the remaining CPU gap.

### Completed NumPy-native voxel preprocessing

PDB parsing and Gaussian residue-density cache construction now remain NumPy-native. The host
voxelizer normalizes legacy JAX inputs once at entry and converts only the completed voxel and
corner to JAX for the bbox pipeline. This removes per-atom device-to-host conversions without
introducing a variable-shape JAX scatter kernel.

The 584-residue 6NT5 microprofile fell from 60.25 ms with JAX coordinates and boxes, or 15.87 ms
with NumPy coordinates and JAX boxes, to approximately 3.6 ms with host-owned preprocessing at
default precision. The synchronized x64 stage measures 5.84 ms versus 7.33 ms upstream.

Results:

- Voxelization: 19.40 ms to 5.84 ms, a 69.9% reduction.
- JAX setup: 1.12 s to 80.58 ms, a 92.8% reduction.
- Warm full pipeline: 46.58 ms to 36.07 ms, a 22.6% reduction.
- JAX/upstream full-pipeline ratio: 1.90 to 1.53.
- Voxelization now ranks eighth with no positive contribution to the remaining CPU gap.

### Completed fused radius and sphere optimization

Molecular-radius calculation now uses compiled fixed-shape masked reductions instead of dynamic
positive-voxel gathers. Internal descriptor and superposition workflows use a fused kernel that
also constructs normalized bbox samples; the original standalone APIs remain compatible.

The real 6NT5 x64 microprofile measured 0.439 ms for the checked fused public call and 0.387 ms for
the underlying kernel, versus 9.33 ms for the former eager path. In the alternating full stage
profile, radius and sphere measure 1.52 ms versus 5.23 ms upstream.

Results:

- Radius and sphere: 12.10 ms to 1.52 ms, an 87.5% reduction.
- JAX first execution: 3.09 s to 2.19 s, a 29.1% reduction.
- Warm full pipeline: 36.07 ms to 21.82 ms, a 39.5% reduction.
- JAX/upstream full-pipeline ratio: 1.53 to 1.20.
- Radius and sphere now rank eighth with no positive contribution to the remaining CPU gap.

### Completed direct bbox-moment contraction

Bounding-box moments now use a dtype-aware compiled kernel keyed by voxel shape and static maximum
order. It integrates each coordinate power directly over every voxel cell and contracts the three
one-dimensional bases with the voxel density, removing the padded volume, triple finite
difference, transposes, and eager denominator grid. The public signature and output layout remain
unchanged, including zero mass and NaN center behavior for an empty voxel.

Independent unit coverage checks asymmetric non-unit cell edges against a NumPy cell-integral
reference. The regression suite covers maximum orders 1, 3, 6, 10, and 20 and both runtime float
configurations. Benchmark schema-v3 metadata records the new representation.

Results:

- Order-1 bbox: 4.57 ms to 0.617 ms, an 86.5% reduction and below the 0.5 ms microbenchmark target
  once synchronized stage-wrapper overhead is excluded.
- Maximum-order bbox: 3.96 ms to 0.588 ms, an 85.2% reduction and below the 0.75 ms target.
- Warm full pipeline: 21.82 ms to 12.70 ms, a 41.8% reduction.
- JAX/upstream full-pipeline ratio: 1.20 to 0.74; warmed JAX is now faster for this workload.
- Both bbox stages have no positive contribution to the remaining CPU gap.

### Completed prepared bbox-to-ZM conversion

Bbox-to-Zernike conversion now uses an immutable device cache containing normalized complex
coefficients, CLM values, and zero-based int32 gather/scatter indices. A compiled kernel performs
the gather, coefficient multiplication, scatter-add, structural NaN fill, normalization, reshape,
and scaling in one dispatch. The original six-argument API remains a compatibility wrapper;
internal regression and all CLI workflows prepare once and reuse the cached API.

The order-6 prototype measured 0.013 ms for the compiled kernel versus 0.65 ms for the former eager
path with device arrays and 0.86 ms with NumPy caches. Order 20 uses approximately 9.95 MB of
prepared arrays and measured 1.30 ms warmed in the same microprofile.

Results:

- Bbox to ZM: 1.49 ms to 0.106 ms, a 92.9% reduction.
- Warm full pipeline: 12.70 ms to 12.33 ms, a 2.9% reduction.
- JAX/upstream full-pipeline ratio: 0.74 to 0.68.
- Bbox-to-ZM now contributes only 0.93% of the positive CPU gap.

### Completed fused AB candidate generation

Single and grouped AB generation now fuse parity-specific coefficient construction, the initial
polynomial roots, batched secondary roots, Cayley--Klein construction, and validity masks into one
static-order JIT kernel. The JAX-native representation retains fixed-shape pairs plus a boolean
mask. Compatibility wrappers preserve the dynamically sized JAX array and list-of-NumPy-arrays
contracts by compacting on CPU.

Profiling showed fixed generation at 0.020--0.034 ms, JAX boolean compaction at approximately
0.38 ms, and CPU compaction at 0.020--0.036 ms. The regression pipeline therefore uses the fixed
representation internally and compacts once immediately before the existing rotation API.

Results:

- AB candidates: 1.63 ms to 0.107 ms, a 93.5% reduction.
- Warm full pipeline: 12.33 ms to 10.93 ms, an 11.4% reduction.
- JAX/upstream full-pipeline ratio: 0.68 to 0.60.
- AB candidates are now approximately 2.79 times faster than upstream for this workload.

## JAX/NumPy Boundary Audit

Legacy NumPy return types should remain compatibility wrappers, not the internal computational
representation. JAX-native kernels should retain fixed-shape arrays plus validity masks until an
explicit public or I/O boundary requires conversion.

Resolved boundaries:

1. Internal rotation consumers retain a batched JAX result through invariant calculation or until
   the superposition host boundary; only the legacy public wrapper returns a NumPy list.
2. Rotation's large binomial/CLM caches and index arrays are prepared once and reused.
3. PDB coordinates and residue-density boxes remain NumPy-native through voxel accumulation; the
   completed voxel and corner cross to JAX once.
4. Bbox-to-ZM G/CLM arrays and gather/scatter indices are prepared once and reused by internal
   workflows.
5. AB roots and candidates are generated as fixed-shape JAX pairs plus masks; regression compacts
   once at the rotation boundary rather than using JAX dynamic filtering.

Remaining unnecessary boundaries:

1. `calculate_ab_rotation_all` transfers candidates and masks to NumPy; superposition workflows
   immediately stack them before passing the batch into JAX rotation code.
2. The regression helper converts scaled JAX moments to NumPy before invoking the JAX descriptor,
   slightly inflating the descriptor-stage measurement.
3. CLI descriptor assembly uses NumPy filtering and concatenation on JAX results. This conversion
   should occur once, after JAX-side assembly, at the public output boundary.
4. Superposition constructs transformation matrices in JAX and then transfers them into
   `np.linalg.solve`; this small path should consistently use one array library.

The highest-value remaining work is descriptor assembly.

## Recommended Next Work

### 1. Keep descriptor assembly on device

Descriptor assembly is the largest positive gap at 0.805 ms versus 0.031 ms upstream and accounts
for 96.23% of positive excess. Remove the regression helper's pre-call NumPy copy, compile the
descriptor reductions, and keep CLI filtering and concatenation in JAX until the public output
boundary.

### 2. Use fixed candidates directly in superposition

Superposition still calls the legacy all-orders wrapper, stacks NumPy groups, and transfers the
valid pairs back into JAX rotation. Use the fixed grouped candidate API and compact once at the
explicit transformation-selection boundary. Treat this as boundary cleanup unless profiling shows
a material end-to-end benefit.

### 3. Add a throughput profile

The existing result measures single-protein latency, which favors NumPy and cannot amortize JAX
compilation or dispatch. Add a separate batch-throughput scenario using repeated committed inputs
or a deterministic set of CA traces. Report proteins per second and per-protein latency for batch
sizes 1, 4, and 16. Keep single-protein latency and batch throughput as separate metrics.

### 4. Establish baselines only after optimization

Continue logging timing without a pass/fail threshold until the stage profiler and batch benchmark
are stable. If a performance gate is later added, store baselines by CPU model and compare robust
medians with a generous noise allowance. Numerical parity must remain the only portable regression
gate for now.

## Commands

Run the default correctness suite:

```bash
uv run --no-sync pytest -q
```

Run the order-20 regression tier:

```bash
uv run --no-sync pytest -q ZMPY3D_JAX/tests/integration/test_upstream_regression.py -m slow
```

Run the CPU performance comparison:

```bash
uv run --no-sync pytest -q -m benchmark \
  ZMPY3D_JAX/tests/benchmark/test_benchmark_regression_upstream.py
```

Control timing sample size with `ZMPY3D_REGRESSION_REPEATS` and
`ZMPY3D_REGRESSION_SAMPLES`. Set `ZMPY3D_BENCHMARK_OUTPUT` to redirect structured JSON results.
If the default uv cache is not writable in a restricted environment, prefix commands with
`UV_CACHE_DIR=/tmp/uv_cache`.

## Constraints

- Keep CPU as the default backend for now.
- Keep x64 enabled in scientific regression and performance characterization.
- Do not add order-40 coverage until its cache and runtime budget are deliberately brought into
  scope.
- `externals/ZMPY3D` remains the reference implementation and must be available for these tests.
- Preserve all current per-stage numerical tolerances unless a separately justified accuracy
  investigation supports changing them.
