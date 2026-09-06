"""Mass-normalized Gaussian boxes for an all-atom molecular density."""

from __future__ import annotations

from typing import Mapping

import numpy as np

import ZMPY3D_JAX.config as _config

from .atomic_properties import BONDI_VDW_RADII, STANDARD_ATOMIC_MASSES


# Radius of a unit 3D isotropic normal distribution containing 95% probability.
_CHI3_95_QUANTILE = 2.7954834829151074


def get_atomic_gaussian_density_cache(
    grid_width: float,
    *,
    radius_map: Mapping[str, float] | None = None,
    mass_map: Mapping[str, float] | None = None,
    density_multiplier: float = 100.0,
    sd_cutoff: float = 3.0,
) -> dict[str, np.ndarray]:
    """Create element-specific Gaussian boxes for one voxel width.

    Gaussian sigma is calibrated so 95% of the untruncated 3D density lies
    inside the element's van der Waals radius. Each discretized box is then
    normalized so ``sum(box) * grid_width**3`` equals atomic mass times
    ``density_multiplier``.
    """
    if not np.isfinite(grid_width) or grid_width <= 0:
        raise ValueError("grid_width must be finite and positive")
    if not np.isfinite(sd_cutoff) or sd_cutoff <= 0:
        raise ValueError("sd_cutoff must be finite and positive")
    if not np.isfinite(density_multiplier) or density_multiplier <= 0:
        raise ValueError("density_multiplier must be finite and positive")

    radii = BONDI_VDW_RADII if radius_map is None else radius_map
    masses = STANDARD_ATOMIC_MASSES if mass_map is None else mass_map
    missing = sorted(set(radii) ^ set(masses))
    if missing:
        raise ValueError(f"radius and mass maps must contain the same elements: {', '.join(missing)}")

    boxes: dict[str, np.ndarray] = {}
    for raw_element, raw_radius in radii.items():
        element = raw_element.upper()
        radius = float(raw_radius)
        mass = float(masses[raw_element])
        if radius <= 0 or mass <= 0 or not np.isfinite(radius + mass):
            raise ValueError(f"invalid radius or mass for element {element}")

        sigma = radius / _CHI3_95_QUANTILE
        half_edge = max(1, int(np.ceil(sd_cutoff * sigma / grid_width)))
        axis = np.arange(-half_edge, half_edge + 1, dtype=np.float64) * grid_width
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        box = np.exp(-(x * x + y * y + z * z) / (2.0 * sigma * sigma))
        box *= mass * density_multiplier / (np.sum(box) * grid_width**3)
        boxes[element] = np.asarray(box, dtype=_config.FLOAT_DTYPE)

    return boxes
