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

Batch descriptor calculation uses bounded, input-order device batches. Host voxelization remains
sequential; each chunk is high-side zero-padded to its largest voxel shape, transferred once, and
processed together through descriptor assembly and normalization rotation:

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

Float32 normalization uses deterministic segmented rotation reduction; x64 retains the faster CPU
scatter reduction. The default suite runs the order-6 float32 representation regression on CPU.
Run both order 6 and order 20 on CUDA with:

```bash
env -u LD_LIBRARY_PATH ZMPY3D_FLOAT32_REGRESSION_BACKEND=gpu \
  pytest -m "not benchmark" \
  ZMPY3D_JAX/tests/integration/test_float32_normalization_regression.py
```

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

By default it alternates the committed 6NT5/6NT6 fixtures at batch sizes 1, 4, and 16. Override
these with `ZMPY3D_BATCH_SIZES`, and control sampling with `ZMPY3D_BATCH_REPEATS` and
`ZMPY3D_BATCH_SAMPLES`. The schema-v2 JSON separates host preparation, transfer, first compilation,
warmed device-core throughput, prepared end-to-end throughput, and production Mode 0/1/2 timings.
It also records a synchronized stage profile for moments, 3DZD, each normalization order's AB
candidates, rotation and invariant reduction, and final assembly. These stage timings are
diagnostic and should not be summed to reconstruct fused device-core latency. Results are saved as:

```text
ZMPY3D_JAX/tests/benchmark/_simple_time_benchmark/batched_pipeline_benchmark_*_*.json
```

Use `ZMPY3D_BATCH_BENCHMARK_OUTPUT` to select another output directory. As with the single-protein
harness, results are informational and numerical parity is checked before timing.

Run a focused module test:

```bash
pytest ZMPY3D_JAX/tests/module/test_calculate_bbox_moment.py
```

The upstream submodule is intended for parity checks. A useful next validation target is a golden-test suite that computes fixtures with `externals/ZMPY3D` and compares this package within explicit tolerances.

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
