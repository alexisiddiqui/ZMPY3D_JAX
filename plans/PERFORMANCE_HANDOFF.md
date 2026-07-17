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
Default suite:             131 passed, 68 deselected
Order-20 regression tier:  2 passed
CPU timing benchmark:      3 passed
```

Latest measurement on a 12th Gen Intel Core i7-12700KF:

```text
Shared input/cache setup:   12.25 ms
JAX setup:                  80.58 ms
JAX rotation-cache setup:   11.94 ms
Upstream setup:             75.45 ms
JAX first execution:         3.09 s
Upstream first execution:   21.37 ms
JAX warmed median:          36.07 ms
Upstream warmed median:     23.55 ms
JAX/upstream time ratio:     1.53
```

Latest structured result:
`upstream_regression_benchmark_20260717T111623_287904Z.json`.

Setup and first execution are measured independently. Warmed JAX is about 1.53 times slower than
warmed upstream NumPy for this single-protein CPU workload. Rotation-cache materialization is an
explicit one-time setup cost and is excluded from warmed stage timing.

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
| 1 | Radius and sphere | 12.10 ms | 6.07 ms | 62.54% |
| 2 | Bbox to ZM | 1.41 ms | 0.086 ms | 13.75% |
| 3 | AB candidates | 1.41 ms | 0.316 ms | 11.40% |
| 4 | Descriptor | 0.630 ms | 0.035 ms | 6.19% |

AB candidate generation now measures 1.41 ms versus 0.316 ms upstream, ranks third, and accounts
for 11.40% of the remaining positive CPU gap.

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

Remaining unnecessary boundaries:

1. `calculate_ab_rotation_all` transfers candidates and masks to NumPy; superposition workflows
   immediately stack them before passing the batch into JAX rotation code.
2. Bbox-to-Zernike conversion rematerializes its static G/CLM caches as JAX arrays on each call.
3. The regression helper converts scaled JAX moments to NumPy before invoking the JAX descriptor,
   slightly inflating the descriptor-stage measurement.
4. CLI descriptor assembly uses NumPy filtering and concatenation on JAX results. This conversion
   should occur once, after JAX-side assembly, at the public output boundary.
5. Superposition constructs transformation matrices in JAX and then transfers them into
   `np.linalg.solve`; this small path should consistently use one array library.

The highest-value remaining stage is radius and sphere construction.

## Recommended Next Work

### 1. Profile and optimize radius and sphere construction

Separate the cost of positive-voxel selection, coordinate construction, radius reductions, and
sphere sample construction. Investigate compiling the fixed-shape reductions and replacing dynamic
boolean gathers if they remain the dominant cost. Preserve variable voxel dimensions and current
radius parity.

### 2. Prepare bbox-to-ZM static caches once

Bbox-to-ZM is the second-largest positive gap and rematerializes its G/CLM arrays on each pipeline
call. Apply the prepared-cache pattern used by rotation before changing its numerical kernel.

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
