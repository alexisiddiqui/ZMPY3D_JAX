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
Default suite:             128 passed, 68 deselected
Order-20 regression tier:  2 passed
CPU timing benchmark:      3 passed
```

Latest measurement on a 12th Gen Intel Core i7-12700KF:

```text
Shared input/cache setup:   31.34 ms
JAX setup:                   1.13 s
Upstream setup:             76.51 ms
JAX first execution:         4.95 s
Upstream first execution:   16.50 ms
JAX warmed median:          88.94 ms
Upstream warmed median:     17.33 ms
JAX/upstream time ratio:     5.13
```

Setup and first execution are now measured independently. Warmed JAX is about 5.13 times slower
than warmed upstream NumPy for this single-protein CPU workload. The old result near `1.00x` was
invalid because both timed wrappers executed the same combined JAX-plus-upstream helper.

## Completed Performance Diagnosis

The benchmark now emits schema-v2 JSON with separate shared setup, implementation setup, first
execution, warmed end-to-end latency, and synchronized stage profiles. Ratio fields are explicit:
`jax_over_upstream_time_ratio` and `upstream_over_jax_time_ratio`.

The synchronized stage profile ranks bottlenecks by positive median excess time. It is diagnostic;
stage medians should not be summed or substituted for end-to-end latency.

Latest ranking:

| Rank | Stage | JAX median | Upstream median | Positive-gap contribution |
| ---: | --- | ---: | ---: | ---: |
| 1 | AB candidates | 36.47 ms | 0.317 ms | 50.13% |
| 2 | ZM rotation | 16.26 ms | 0.464 ms | 21.91% |
| 3 | Voxelization | 18.14 ms | 7.49 ms | 14.77% |
| 4 | Radius and sphere | 9.75 ms | 4.99 ms | 6.60% |

AB candidate generation is the next optimization target: it is approximately 115 times slower
than upstream and accounts for half of the measured positive CPU gap.

## Recommended Next Work

### 1. Optimize AB candidate generation

Profile `calculate_ab_rotation` internally, separating polynomial/root solving, candidate
construction, validity filtering, and host conversion. Examine it for:

- Repeated NumPy-to-JAX or JAX-to-NumPy conversions.
- Python loops surrounding small dispatched JAX operations.
- Missing or overly narrow JIT boundaries.
- Recomputed constants or indices that can be cached.
- Opportunities to keep intermediate arrays on the JAX device through adjacent stages.

After each change, rerun the upstream regression suite before accepting the performance result.

### 2. Reassess rotation after AB optimization

Rerun the full stage profile after improving AB generation. If ZM rotation remains the largest
positive contributor, create a separate optimization plan for its compatibility wrapper and device
transfer behavior.

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
