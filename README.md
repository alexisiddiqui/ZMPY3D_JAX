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

Run the informational CPU performance comparison with independent JAX and upstream pipelines:

```bash
pytest -m benchmark ZMPY3D_JAX/tests/benchmark/test_benchmark_regression_upstream.py
ZMPY3D_REGRESSION_REPEATS=5 ZMPY3D_REGRESSION_SAMPLES=9 \
  pytest -m benchmark ZMPY3D_JAX/tests/benchmark/test_benchmark_regression_upstream.py
```

The benchmark enables CPU/x64 explicitly, reports cold JAX and warmed timing distributions, and
writes timestamped JSON results under:

```text
ZMPY3D_JAX/tests/benchmark/_simple_time_benchmark/upstream_regression_benchmark_*.json
```

Set `ZMPY3D_BENCHMARK_OUTPUT` to write results to another directory. Timing is deliberately
informational and does not fail on a hardware-dependent speed threshold.

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
