# ZMPY3D_JAX

JAX-oriented fork of [ZMPY3D](https://github.com/tawssie/ZMPY3D), a Python implementation of 3D Zernike moments for protein structure volume analysis.

This repository keeps the structural-biology workflow from the original project: parse PDB CA traces, voxelize residues with Gaussian density boxes, compute geometric moments, convert them to 3D Zernike moments, derive rotation-invariant descriptors, and support shape comparison/superposition workflows.

The original NumPy implementation is included as a git submodule at:

```text
externals/ZMPY3D
```

Use it as the numerical reference while completing and validating this JAX port.

## Status

This is not yet a fully validated general-purpose Zernike moments library.

Implemented or partially implemented:

- Legacy PDB CA-trace parsing for protein residue records.
- Residue Gaussian density voxelization.
- Bounding-box/geometric moment calculation.
- Bounding-box moment to Zernike moment conversion.
- 3DZD invariant descriptor generation.
- Canterakis-style normalizing rotation workflows.
- CLI entry points for ZM descriptors, shape score, batch descriptors, and superposition.
- JAX conversions for several numerical kernels.

Known gaps:

- Some rotation-normalization code still uses NumPy and Python loops.
- Rotation normalization retains a compatibility wrapper around a vectorized JAX kernel.
- Input handling is intentionally structural-biology-specific; arbitrary point clouds are not a public API.
- Core numerical stages have parity tests against the upstream NumPy implementation.
- Max-order 40 cache data is not included in the repository because of size.

## Installation

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
cd ZMPY3D_JAX
```

If the repository was already cloned:

```bash
git submodule update --init --recursive
```

Install in editable mode:

```bash
python -m pip install -e .
```

Runtime requirements are declared in `pyproject.toml`.

### NVIDIA GPU / CUDA 13

On a Linux system with a supported NVIDIA GPU and current driver, install JAX's bundled CUDA 13
runtime into the project's `uv` environment:

```bash
env -u LD_LIBRARY_PATH uv pip install --upgrade "jax[cuda13]"
```

JAX recommends unsetting `LD_LIBRARY_PATH` when using its bundled CUDA wheels because locally
installed CUDA libraries can override the wheel-provided versions. Use the same environment when
running GPU commands, or remove the conflicting CUDA paths from your shell configuration.

Verify that JAX can see the GPU:

```bash
env -u LD_LIBRARY_PATH uv run --no-sync python -c 'import jax; print(jax.default_backend()); print(jax.devices())'
```

The output should report the `gpu` backend and at least one `CudaDevice`. Configure this package
for GPU execution before creating JAX arrays or invoking numerical kernels:

```bash
env -u LD_LIBRARY_PATH uv run --no-sync python -c 'import ZMPY3D_JAX as z; z.configure_for_scientific_computing(enable_x64=True, platform="gpu"); import jax; print(jax.devices())'
```

In Python applications, call:

```python
import ZMPY3D_JAX as z

z.configure_for_scientific_computing(enable_x64=True, platform="gpu")
```

The project's default remains CPU, and CUDA-enabled JAX wheels are currently installed separately
rather than being required for all package users.

## Quick Start

Compute descriptors from a PDB file:

```bash
ZMPY3D_CLI_ZM 6NT5.pdb 1.0 6 5 2
```

Arguments:

- `PDBFile`: input `.pdb` or `.txt` file in legacy PDB text format.
- `GridWidth`: voxel width; currently accepted CLI values are `0.25`, `0.50`, or `1.00`.
- `MaximumOrder`: currently accepted CLI values are `6`, `20`, or `40`.
- `NormOrder`: maximum normalization order, from `2` to `MaximumOrder`.
- `Mode`: `0` for Canterakis normalization, `1` for 3DZD invariant, `2` for both.

Use the Python API:

```python
import ZMPY3D_JAX as z

z.configure_for_scientific_computing(enable_x64=True)  # CPU is the current default

descriptor = z.ZMPY3D_CLI_ZM(
    "6NT5.pdb",
    GridWidth=1.0,
    MaxOrder=6,
    MaxTargetOrder2NormRotate=5,
    Mode=2,
)

# Device-native, fixed-shape output. Invalid numerical slots are retained in-place.
descriptor.values
descriptor.is_valid
```

`ZMPY3D_CLI_ZM` returns a `DescriptorVector` whose `values` and `is_valid` fields are
JAX arrays. Structural holes are removed with fixed gather indices, while data-dependent
invalid values remain represented by the mask so descriptor assembly can stay JIT/GPU compatible.
`ZMPY3D_CLI_BatchZM` returns the same type with a leading batch dimension. Shape-score APIs
likewise return JAX scalar arrays (or stacked one-dimensional arrays for batch calls). Console
entry points transfer and compact results only when printing.

Batch descriptor calculation uses bounded, input-order device batches. Production host preparation
remains sequential pending the prefetch promotion gate. The internal benchmark can compare it with
a bounded two-chunk thread-pool prototype that preserves input order and propagates worker errors:

```python
batch = z.ZMPY3D_CLI_BatchZM(
    ["6NT5.pdb", "6NT6.pdb"],
    GridWidth=1.0,
    MaxOrder=6,
    MaxTargetOrder2NormRotate=5,
    Mode=2,
    BatchSize=16,
)
```

`BatchSize` defaults to 16; lower it when processing unusually large voxel grids or higher moment
orders. The console command accepts the equivalent optional flag:

```bash
ZMPY3D_CLI_BatchZM pdb_files.txt 1.0 6 5 2 --batch-size 16
```

### Experimental all-atom density

The Python API also provides an explicit heavy-atom representation. It places a
mass-normalized Gaussian at every selected atom, derives Gaussian width from a
Bondi-style elemental van der Waals radius, and scales density by the PDB occupancy:

```python
import ZMPY3D_JAX as z

descriptor = z.ZMPY3D_CLI_ZM(
    "6NT5.pdb",
    GridWidth=1.0,
    MaxOrder=6,
    MaxTargetOrder2NormRotate=5,
    Mode=2,
    Representation="all_atom_gaussian",
)
```

The default selection includes heavy `ATOM` records from model 1 across all chains.
Hydrogens, `HETATM` records, and waters are excluded. Selection can be made explicit:

```python
descriptor = z.ZMPY3D_CLI_ZM(
    "complex.cif",
    Representation="all_atom_gaussian",
    ChainID="A",
    Model=1,
    AssemblyID="1",
    IncludeHetero=True,
    IncludeWater=False,
    IncludeHydrogens=False,
)
```

The new all-atom path uses Biotite for legacy PDB and mmCIF input, including
occupancy-based alternate-location selection and optional biological assemblies.
It supports the organic and halogen elements in the packaged atomic-property table; unsupported
elements, including metals without an approved radius policy, raise `ValueError`.
All-atom descriptors and the original `ca_residue` descriptors are different feature
spaces and must not be compared with each other. The original APIs continue to use
`ca_residue` unchanged.

Heterogeneous files and multi-model ensembles can be processed together. Rows retain
their structure IDs and representation, and malformed samples can either abort or be
recorded and skipped:

```python
batch = z.calculate_structure_descriptors_batch(
    ["protein.pdb", "ensemble.cif"],
    representation="all_atom_gaussian",
    model=None,
    mode=1,
    on_error="skip",
)

batch.ids
batch.representation
batch.values
batch.failures
```

The heterogeneous runner queues at most four voxel budgets of prepared grids at a
time, packs compatible shapes by their incremental padded cost under the device voxel
budget, and rounds padded dimensions to multiples of eight. Partial chunks use their
actual sample count; a structure larger than the budget runs alone. Order-20 batches
use a strict float64 internal pipeline by default, including normalization, and cast
only the completed descriptor back to the configured public dtype.

## Cache Data

The repository includes cache files for lower-order workflows:

```text
ZMPY3D_JAX/cache_data/BinomialCache.pkl
ZMPY3D_JAX/cache_data/LogG_CLMCache_MaxOrder06.pkl
ZMPY3D_JAX/cache_data/LogG_CLMCache_MaxOrder20.pkl
```

The max-order 40 cache is large and should be downloaded separately when needed. The original project documents the order-40 cache source; place the resulting file here:

```text
ZMPY3D_JAX/cache_data/LogG_CLMCache_MaxOrder40.pkl
```

## Development

Run the offline correctness suite (benchmarks are excluded by default):

```bash
pytest
```

Run performance tests separately:

```bash
pytest -m benchmark
```

The default suite includes fast, end-to-end numerical regression cases against the upstream
NumPy implementation. Run them directly with:

```bash
pytest ZMPY3D_JAX/tests/integration/test_upstream_regression.py -m "not slow"
```

Run the slower order-20 cases separately:

```bash
pytest ZMPY3D_JAX/tests/integration/test_upstream_regression.py -m slow
```

Float32 moment conversion and normalization use deterministic segmented reductions; x64 retains
the scatter reductions. The default suite runs order-6 float32 representation and stage-boundary
repeatability regressions on CPU. Run both order 6 and order 20 on CUDA with:

```bash
env -u LD_LIBRARY_PATH ZMPY3D_FLOAT32_REGRESSION_BACKEND=gpu \
  pytest -m "not benchmark" \
  ZMPY3D_JAX/tests/integration/test_float32_normalization_regression.py \
  ZMPY3D_JAX/tests/integration/test_float32_stage_determinism.py \
  ZMPY3D_JAX/tests/integration/test_float32_structure_accuracy.py
```

The structure-accuracy regression compares float32 against an isolated x64 CPU reference and
reports both per-structure error and preservation of the `6NT5`–`6NT6` descriptor difference.
Set `ZMPY3D_FLOAT32_ACCURACY_OUTPUT` to a directory to write schema-v2 JSON reports for orders 6
and 20.

Run the internal order-20 mixed-precision timing prototype separately on CPU and GPU:

```bash
ZMPY3D_BENCHMARK_BACKEND=cpu pytest -q -m benchmark \
  ZMPY3D_JAX/tests/benchmark/test_benchmark_mixed_precision.py

env -u LD_LIBRARY_PATH ZMPY3D_BENCHMARK_BACKEND=gpu pytest -q -m benchmark \
  ZMPY3D_JAX/tests/benchmark/test_benchmark_mixed_precision.py
```

Float32 batched descriptors now select the winning mixed-moment frontier automatically at order 20
and above: Cartesian moments and bbox-to-ZM conversion use x64, then 3DZD and normalization return
to float32. Order 6 and fully x64 execution are unchanged. Internal benchmarks retain a
`moment_precision="configured"` override for baseline comparison.

Run the informational CPU performance comparison with independent JAX and upstream pipelines:

```bash
pytest -m benchmark ZMPY3D_JAX/tests/benchmark/test_benchmark_regression_upstream.py
ZMPY3D_REGRESSION_REPEATS=5 ZMPY3D_REGRESSION_SAMPLES=9 \
  pytest -m benchmark ZMPY3D_JAX/tests/benchmark/test_benchmark_regression_upstream.py
```

After installing the CUDA 13 JAX wheels, run the same synchronized harness on GPU in a separate
process:

```bash
env -u LD_LIBRARY_PATH ZMPY3D_BENCHMARK_BACKEND=gpu \
  uv run --no-sync pytest -q -m benchmark \
  ZMPY3D_JAX/tests/benchmark/test_benchmark_regression_upstream.py
```

`ZMPY3D_BENCHMARK_BACKEND` defaults to `cpu`. Run CPU and GPU trials separately because JAX backend
selection is process-global. Both results record the selected backend and device list in JSON;
filenames include `_cpu_` or `_gpu_` for direct comparison. The upstream side remains NumPy/CPU in
both trials. The harness verifies numerical parity before collecting warmed samples. Its
synchronized per-stage profile deliberately penalizes very small GPU kernels with launch overhead;
use the warmed end-to-end result for the primary single-protein comparison.

The benchmark enables CPU/x64 explicitly and writes schema-v4 JSON containing separate setup,
first-execution, warmed end-to-end, and synchronized per-stage timing distributions. It also ranks
stages by their contribution to the warmed JAX-over-upstream CPU gap. Results are written under:

```text
ZMPY3D_JAX/tests/benchmark/_simple_time_benchmark/upstream_regression_benchmark_*_*.json
```

Set `ZMPY3D_BENCHMARK_OUTPUT` to write results to another directory. Timing is deliberately
informational and does not fail on a hardware-dependent speed threshold. Synchronized stage
timings are diagnostic and should not be summed or treated as end-to-end latency.

Run the production batch-throughput harness separately on CPU and GPU:

```bash
ZMPY3D_BENCHMARK_BACKEND=cpu \
  uv run --no-sync pytest -q -m benchmark \
  ZMPY3D_JAX/tests/benchmark/test_benchmark_batched_pipeline.py

env -u LD_LIBRARY_PATH ZMPY3D_BENCHMARK_BACKEND=gpu \
  uv run --no-sync pytest -q -m benchmark \
  ZMPY3D_JAX/tests/benchmark/test_benchmark_batched_pipeline.py
```

By default it profiles the production float32/mixed-precision order-20 path with alternating
6NT5/6NT6 fixtures at batch sizes 2 and 16. Set `ZMPY3D_BATCH_INPUT_MANIFEST` to a text file
containing one PDB path per line to profile a heterogeneous production workload. Override the
batch sizes with `ZMPY3D_BATCH_SIZES` and the order with `ZMPY3D_BATCH_MAX_ORDER`, and control
sampling with `ZMPY3D_BATCH_REPEATS` and `ZMPY3D_BATCH_SAMPLES`. The schema-v8 JSON separates
parse, host accumulation, padding, synchronized batched transfer, first compilation,
warmed device-core throughput, whole-pipeline JIT throughput, prepared end-to-end throughput, and
production Mode 0/1/2 timings. It records both fused normalization methods and a diagnostic split
of compact AB candidates, rotation, invariant reduction, moments, 3DZD, and assembly. Candidate
solver probes compare per-order and degree-grouped companion layouts, nested and flattened
rotation, and the benchmark-only analytic odd-order experiment. Stage timings are synchronized
diagnostics and should not be
summed to reconstruct fused device-core latency.

Set `ZMPY3D_BATCH_TRACE_DIR` to capture one warmed separate-pipeline and whole-JIT execution for
the largest configured batch using JAX profiler annotations. The same benchmark command can be
launched under Nsight Systems when a CUDA kernel timeline is required.
Results are saved as:

```text
ZMPY3D_JAX/tests/benchmark/_simple_time_benchmark/batched_pipeline_profile_{cpu,gpu}.json
```

Use `ZMPY3D_BATCH_BENCHMARK_OUTPUT` to select another output directory. As with the single-protein
harness, results are informational and numerical parity is checked before timing.

Production compact normalization uses companion-matrix eigensolves for every initial polynomial.
The exact secondary quartic factorization retains only its two useful real roots. The analytic
odd-order quadratic remains an explicit benchmark experiment, not a production default. The
prepared batch CLI compiles the complete device descriptor path once and reuses it across chunks
and compatible input shapes.

Production uses degree-grouped companion normalization with nested rotation. The schema-v7
promotion measurements on an RTX 3090 found degree-grouped companion normalization
`7.02%` and `7.26%` faster than per-order companion normalization at batch 16 across two clean
runs. It also improved GPU batch 2 and CPU batch 2, while CPU batch 16 was effectively neutral
(`0.13%` slower, within the 10% guard). Nested rotation remains preferred: the flattened prototype
was slower on both GPU and CPU. Host prefetch improved prepared throughput by at most `1.38%` on
GPU and `0.71%` on CPU, below its 10% gate, so production host preparation remains sequential.

Isolated batch-16 CUDA allocator measurements recorded `303,990,784` bytes peak for per-order
companion normalization and `378,566,400` bytes for degree grouping. The `24.53%` increase passes
the 25% memory-growth gate.

The focused CUDA regression suite passed all six tests. Order-20 6NT5/6NT6 retained score error
`0.00294755`, difference cosine `0.99976075`, separation ratio `0.99944251`, candidate counts
`8/4/8/4`, and bitwise repeatability over five direct and whole-JIT executions.
Those figures predate the strict order-20 precision policy and do not establish heterogeneous
padding invariance; use the current padding regression and collect fresh target-hardware
throughput measurements before treating them as production order-20 results.

Run a focused module test:

```bash
pytest ZMPY3D_JAX/tests/module/test_calculate_bbox_moment.py
```

The upstream submodule is intended for parity checks. A useful next validation target is a golden-test suite that computes fixtures with `externals/ZMPY3D` and compares this package within explicit tolerances.

### Pinned TensorFlow comparison benchmark

The repository also includes the pinned TensorFlow reference at `externals/ZMPY3D_TF`:

```bash
git submodule update --init --recursive
uv sync --project ZMPY3D_JAX/tests/benchmark/tensorflow_env
```

Run the optional benchmark in a separate process for each framework. It reports both the production
JAX mixed-precision view and an x64-matched view against TensorFlow 2.20:

```bash
env -u LD_LIBRARY_PATH ZMPY3D_TF_BENCHMARK_BACKEND=gpu \
  ZMPY3D_TF_BENCHMARK_ORDERS=6,20 ZMPY3D_TF_BENCHMARK_BATCHES=1,2,16 \
  ZMPY3D_TF_BENCHMARK_SAMPLES=3 ZMPY3D_TF_BENCHMARK_REPEATS=2 \
  uv run --no-sync pytest -q -m benchmark \
  ZMPY3D_JAX/tests/benchmark/test_benchmark_tensorflow.py
```

Set `ZMPY3D_TF_BENCHMARK_BACKEND=cpu` for CPU and `ZMPY3D_TF_PYTHON` when the TensorFlow
environment is elsewhere. Singular `ZMPY3D_TF_BENCHMARK_ORDER` and `ZMPY3D_TF_BENCHMARK_BATCH`
controls remain available for smoke runs. Each repeat launches fresh framework processes; each
sample is a synchronized post-warmup measurement inside its worker. The TensorFlow reference
processes structures independently through its `tf.data` generator, whereas JAX uses true padded
device batches. Compare core timings as reference-implementation throughput and treat public
end-to-end timings as workflow measurements rather than interchangeable isolated kernels. Reports
include every descriptor row, compile time, framework/device metadata, error metrics, and are
written under `ZMPY3D_JAX/tests/benchmark/_tensorflow` or the directory selected by
`ZMPY3D_TF_BENCHMARK_OUTPUT`.

## Package Layout

```text
ZMPY3D_JAX/
  cache_data/      precomputed binomial/G/CLM caches
  lib/             numerical kernels and IO helpers
  tests/           module, integration, and benchmark tests
externals/ZMPY3D/  upstream NumPy reference implementation
```

## Citation

If you use this work, cite the original ZMPY3D paper:

Lai, J. S., Burley, S. K., & Duarte, J. M. (2024). ZMPY3D: Accelerating protein structure volume analysis through vectorized 3D Zernike moments and Python-based GPU integration. Bioinformatics Advances, vbae111. https://doi.org/10.1093/bioadv/vbae111

## License

This fork retains the repository license in `LICENSE`. Check the upstream submodule for its own license and attribution details.
