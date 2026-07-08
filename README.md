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

- PDB CA-trace parsing for protein/nucleic-acid residue records.
- Residue Gaussian density voxelization.
- Bounding-box/geometric moment calculation.
- Bounding-box moment to Zernike moment conversion.
- 3DZD invariant descriptor generation.
- Canterakis-style normalizing rotation workflows.
- CLI entry points for ZM descriptors, shape score, batch descriptors, and superposition.
- JAX conversions for several numerical kernels.

Known gaps:

- Some rotation-normalization code still uses NumPy and Python loops.
- `calculate_ab_rotation_all` currently needs additional JAX work and validation.
- Input handling is structural-biology-specific; arbitrary volumes and general point clouds need a separate public API.
- Tests need golden-value comparisons against the upstream NumPy implementation and analytic/simple-shape fixtures.
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

z.configure_for_scientific_computing(enable_x64=True, platform="cpu")

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

Run tests:

```bash
pytest
```

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
