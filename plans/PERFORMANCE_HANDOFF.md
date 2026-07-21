# ZMPY3D JAX Regression and Performance Handoff

Date: 2026-07-21

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
Default suite:             178 passed, 74 deselected
Order-20 regression tier:  5 passed
CUDA float32 regression:   4 passed
CPU timing benchmark:      3 passed
```

Latest measurement on a 12th Gen Intel Core i7-12700KF:

```text
Shared input/cache setup:   12.23 ms
JAX setup:                  79.57 ms
JAX rotation-cache setup:   11.50 ms
JAX bbox-to-ZM cache setup: 48.62 ms
Upstream setup:             70.17 ms
JAX first execution:         0.70 s
Upstream first execution:   16.89 ms
JAX warmed median:          10.56 ms
Upstream warmed median:     17.79 ms
JAX/upstream time ratio:     0.59
```

Latest structured result:
`upstream_regression_benchmark_20260721T112442_790371Z.json`.

Setup and first execution are measured independently. Warmed JAX is about 1.68 times faster than
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
| 1 | ZM rotation | 0.521 ms | 0.453 ms | 63.83% |
| 2 | Bbox to ZM | 0.118 ms | 0.085 ms | 31.26% |
| 3 | Descriptor | 0.039 ms | 0.034 ms | 4.91% |
| 4 | AB candidates | 0.124 ms | 0.320 ms | 0% |

AB candidate generation now measures 0.124 ms versus 0.320 ms upstream and has no positive
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

### Completed device-native descriptor assembly

The 3DZD reduction now runs as a compiled kernel, and the regression pipeline no longer copies
scaled JAX moments through NumPy before descriptor calculation. Descriptor-producing APIs return
fixed-shape `DescriptorVector` pytrees containing structurally compact device values plus boolean
validity masks. ZM and ShapeScore assembly, weighting, and score reductions remain in JAX;
data-dependent compaction occurs only at console output boundaries.

Results:

- Descriptor stage: 0.805 ms to 0.039 ms, a 95.1% reduction.
- Descriptor positive-gap contribution: 96.23% to 4.91%.
- JAX first execution: 0.88 s to 0.70 s.
- Warm full pipeline: 10.93 ms to 10.56 ms.
- JAX/upstream full-pipeline ratio: 0.60 to 0.59.

### Completed production device batching

`ZMPY3D_CLI_BatchZM` now processes bounded input-order chunks instead of looping over complete
single-protein descriptor pipelines. Voxelization remains NumPy-native, but host voxels are
high-side padded, stacked, and transferred once per chunk. Bbox moments, radius/sphere samples,
bbox-to-ZM conversion, 3DZD reduction, fixed AB candidates, candidate rotations, masked invariant
means, and descriptor assembly all batch across proteins on device.

The Python API and console command expose a default batch size of 16. Mixed 6NT5/6NT6 parity is
covered for all descriptor modes, partial chunks, and maximum orders 6 and 20. The schema-v2
throughput harness reports host preparation, transfer, first compilation, warmed device-core,
prepared end-to-end, Mode 0/1/2, and synchronized per-stage results for batch sizes 1, 4, and 16.

## JAX/NumPy Boundary Audit

JAX-native kernels retain fixed-shape arrays plus validity masks until an explicit I/O boundary
requires conversion. Descriptor Python APIs are device-first; shell entry points compact on host
only for printing.

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
6. Descriptor reductions, structural gathering, CLI assembly, and ShapeScore reductions stay in
   JAX. The regression helper sends JAX moments directly into the compiled descriptor kernel.

Remaining unnecessary boundaries:

1. `calculate_ab_rotation_all` transfers candidates and masks to NumPy; superposition workflows
   immediately stack them before passing the batch into JAX rotation code.
2. Superposition constructs transformation matrices in JAX and then transfers them into
   `np.linalg.solve`; this small path should consistently use one array library.

The highest-value remaining boundary cleanup is superposition.

## CPU/GPU Latency Comparison

A matched 5-repeat, 9-sample run on JAX/JAXlib 0.11.0 compared the CPU backend with an NVIDIA
GeForce RTX 3090 using CUDA 13. Both runs used x64, the committed 6NT5 input, normalization order
5, explicit synchronization, and the same numerical-parity checks.

| Measurement | JAX CPU | JAX GPU | GPU/CPU time ratio |
| --- | ---: | ---: | ---: |
| First execution | 679.34 ms | 2074.47 ms | 3.05 |
| Warmed end-to-end | 9.08 ms | 11.59 ms | 1.28 |
| Radius and sphere | 1.482 ms | 0.261 ms | 0.18 |
| Maximum-order bbox | 0.465 ms | 0.216 ms | 0.46 |
| AB candidates | 0.129 ms | 1.285 ms | 9.98 |
| ZM rotation | 0.370 ms | 1.985 ms | 5.36 |

The RTX 3090 is approximately 27.7% slower end-to-end for this single-protein workload. It
accelerates the larger reduction/contraction stages, but host voxelization remains unchanged and
small AB/rotation kernels are dominated by GPU launch and synchronization overhead. This result
does not predict batch throughput. The production batch harness is now available; a matched
CPU/GPU stage profile now identifies normalization as the GPU bottleneck.

## Mask-Aware Normalization Result

The secondary candidate quartic factors exactly as
`(t² + 1)(coef4*t² + coef3*t - coef4)`. The `±i` roots are always rejected by
the real-root mask. Production normalization now solves the remaining quadratic directly with a
stable formula, retaining fixed device shapes while reducing capacity from 16 to 8 slots for even
orders and from 8 to 4 slots for odd orders. No per-protein host compaction or bucketing is used.

The schema-v4 harness compares the legacy full-fixed representation, compact per-order execution,
and compact parity-fused execution. A clean batch-16 promotion run (5 samples, 2 repeats, x64,
mixed 6NT5/6NT6 inputs) measured:

| Representation | CPU ms/protein | GPU ms/protein | GPU change vs full |
| --- | ---: | ---: | ---: |
| Full fixed | 2.126 | 18.938 | baseline |
| Analytic compact, per order | 2.009 | 7.797 | 58.8% faster |
| Analytic compact, parity fused | 1.926 | 10.029 | 47.0% faster |

The compact per-order representation remains the production default because it is fastest on the
RTX 3090 and improves on full-fixed CPU execution. The parity-fused variant remains in the harness
as an experimental comparison; CPU differences are small and variable, while grouping even and
odd orders consistently loses to the smaller independent launches on the GPU.
The legacy quartic path remains available as the numerical oracle.

Structured results:
`batched_pipeline_benchmark_cpu_20260721T141713_179373Z.json` and
`batched_pipeline_benchmark_gpu_20260721T141820_460543Z.json`.

### Float32 deterministic rotation

GPU float32 order-20 representation comparisons were initially contaminated by nondeterministic
atomic accumulation in `z_nlm.at[s_id].add`. Candidate pairs were stable to approximately `3e-7`,
but repeated rotations from identical inputs varied by approximately `5e-4`, which amplified to
multi-unit descriptor changes at order 20.

Rotation now provides a deterministic segmented associative reduction for float32 while retaining
scatter for x64. With one frozen raw-moment tensor, five repeated CPU and GPU normalization runs
are bitwise identical. At order 20, all three representations agree within `1.19e-3` absolute on
the RTX 3090 and retain identical masks and candidate-valid counts.

The schema-v4 batch-16 x64 profile measured scatter versus segmented reduction at 2.012 versus
2.508 ms/protein on CPU and 7.857 versus 7.861 ms/protein on GPU. The CPU regression prevents a
global promotion. At order 20, segmented and scatter x64 results differ by at most `6.4e-14` on
CPU/GPU. For float32 order 20 at GPU batch size 2, segmented reduction improved rotation
normalization from 8.50 to 1.77 ms/protein while making it deterministic. Automatic selection is
therefore segmented for float32/complex64 and scatter for x64/complex128.

### Float32 deterministic bbox-to-ZM conversion

Stage-boundary repetition isolated the remaining full-pipeline GPU variation to
`bbox_to_zm`: order-6 Cartesian bbox moments were bitwise repeatable, but converting the same
frozen moments repeatedly varied by up to `2.91e-5`. The cause was the second indexed atomic
accumulation, `summed.at[output_indices].add(contributions)`.

Bbox-to-ZM conversion now shares the sorted associative segmented reduction used by rotation.
Automatic selection uses it for float32/complex64 and retains scatter for x64/complex128. Both
order-6 and order-20 GPU stage tests are bitwise repeatable, including five complete descriptor
pipeline executions from the same voxel batch. The new regression independently freezes inputs at
the order-1 bbox, radius/sample, max-order bbox and bbox-to-ZM boundaries, so future failures name
the first unstable kernel rather than only reporting final descriptor drift.

The schema-v5 benchmark adds a bbox-to-ZM reduction profile. A batch-16 CPU x64 diagnostic measured
scatter at `1.878` and segmented at `1.865` ms/protein for the complete 3DZD-only path (no
meaningful change). A smaller batch-2 GPU x64 diagnostic measured `0.316` versus `0.345`
ms/protein; this short run is informational and production x64 continues to select scatter.

## Previous Batched Stage Profile

The clean, sequentially executed schema-v2 profile used 5 repeats, 9 samples, x64, mixed
6NT5/6NT6 batches, and JAX 0.11.0 on the CPU and RTX 3090. No competing CUDA compute process was
present. At batch size 16:

| Measurement | CPU ms/protein | GPU ms/protein | GPU/CPU ratio |
| --- | ---: | ---: | ---: |
| Mode 1: 3DZD only | 1.776 | 0.102 | 0.06 |
| Mode 0: normalization only | 2.178 | 19.886 | 9.13 |
| Moments category (synchronized) | 2.646 | 0.168 | 0.06 |
| AB candidates (orders 2–5) | 0.079 | 6.270 | 79.36 |
| ZM rotation (orders 2–5) | 0.513 | 13.872 | 27.05 |
| Masked invariant means | 0.039 | 0.064 | 1.65 |

Structured results:
`batched_pipeline_benchmark_cpu_20260721T123436_583183Z.json` and
`batched_pipeline_benchmark_gpu_20260721T123824_171608Z.json`.

GPU rotation takes approximately 4.64 ms/protein for each 16-slot even order and 2.31 ms/protein
for each 8-slot odd order. Exactly half the slots are valid in every order, so the near-linear
slot scaling indicates that invalid fixed slots materially contribute to rotation cost. Candidate
generation shows the same even/odd scaling and does not improve per-protein throughput as the batch
grows. The next GPU optimization should therefore target fixed candidate generation and mask-aware
rotation execution, not voxelization.

## Recommended Next Work

### 1. Use fixed candidates directly in superposition

Superposition still calls the legacy all-orders wrapper, stacks NumPy groups, and transfers the
valid pairs back into JAX rotation. Use the fixed grouped candidate API and compact once at the
explicit transformation-selection boundary. Treat this as boundary cleanup unless profiling shows
a material end-to-end benefit.

### 2. Profile the remaining compact normalization kernels

The invalid-slot work has been removed. Re-profile compact candidate generation and compact
rotation separately before changing voxelization. The next normalization optimization should be
guided by the new compact-stage split rather than by the legacy full-fixed stage totals.

### 3. Establish baselines only after optimization

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

Run the isolated float32 representation regression on CUDA:

```bash
env -u LD_LIBRARY_PATH ZMPY3D_FLOAT32_REGRESSION_BACKEND=gpu \
  uv run --no-sync pytest -q -m "not benchmark" \
  ZMPY3D_JAX/tests/integration/test_float32_normalization_regression.py
```

Run the CPU performance comparison:

```bash
uv run --no-sync pytest -q -m benchmark \
  ZMPY3D_JAX/tests/benchmark/test_benchmark_regression_upstream.py
```

Run the same harness with JAX on GPU and upstream NumPy on CPU:

```bash
env -u LD_LIBRARY_PATH ZMPY3D_BENCHMARK_BACKEND=gpu \
  uv run --no-sync pytest -q -m benchmark \
  ZMPY3D_JAX/tests/benchmark/test_benchmark_regression_upstream.py
```

Run the production batch-throughput harness on either backend:

```bash
ZMPY3D_BENCHMARK_BACKEND=cpu uv run --no-sync pytest -q -m benchmark \
  ZMPY3D_JAX/tests/benchmark/test_benchmark_batched_pipeline.py

env -u LD_LIBRARY_PATH ZMPY3D_BENCHMARK_BACKEND=gpu \
  uv run --no-sync pytest -q -m benchmark \
  ZMPY3D_JAX/tests/benchmark/test_benchmark_batched_pipeline.py
```

Control timing sample size with `ZMPY3D_REGRESSION_REPEATS` and
`ZMPY3D_REGRESSION_SAMPLES`. Set `ZMPY3D_BENCHMARK_OUTPUT` to redirect structured JSON results.
The batch harness uses `ZMPY3D_BATCH_REPEATS`, `ZMPY3D_BATCH_SAMPLES`, and
`ZMPY3D_BATCH_SIZES`.
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
