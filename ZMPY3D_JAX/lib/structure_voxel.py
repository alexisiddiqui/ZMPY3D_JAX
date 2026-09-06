"""Shared structure-to-voxel preparation for single and batch APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .fill_voxel_by_weight_density04 import fill_voxel_by_weight_density_host
from .get_pdb_xyz_ca02 import get_pdb_xyz_ca02
from .structure_io import load_atomic_structures


def voxelize_structure(
    file_name: str,
    *,
    representation: str,
    grid_width: float,
    weight_map: Mapping[str, float],
    density_boxes: Mapping[str, np.ndarray],
    model: int | None = 1,
    chain_ids: Sequence[str] | str | None = None,
    assembly_id: str | None = None,
    include_hetero: bool = False,
    include_water: bool = False,
    include_hydrogens: bool = False,
) -> list[tuple[str, np.ndarray]]:
    """Return identified voxel grids for one file and representation."""
    if representation == "ca_residue":
        xyz, labels = get_pdb_xyz_ca02(file_name)
        samples: list[tuple[str, Any, Sequence[str], Any]] = [
            (f"{Path(file_name)}:ca_residue", xyz, labels, None)
        ]
    elif representation == "all_atom_gaussian":
        samples = [
            (sample.id, sample.xyz, sample.elements, sample.occupancies)
            for sample in load_atomic_structures(
                file_name,
                model=model,
                chain_ids=chain_ids,
                assembly_id=assembly_id,
                include_hetero=include_hetero,
                include_water=include_water,
                include_hydrogens=include_hydrogens,
            )
        ]
    else:
        raise ValueError("representation must be 'ca_residue' or 'all_atom_gaussian'")

    voxels = []
    for sample_id, xyz, labels, multipliers in samples:
        if xyz.shape[0] == 0:
            raise ValueError(f"No atoms selected for {sample_id}")
        voxel, _ = fill_voxel_by_weight_density_host(
            xyz,
            labels,
            dict(weight_map),
            grid_width,
            dict(density_boxes),
            point_weight_multipliers=multipliers,
        )
        voxels.append((sample_id, voxel))
    return voxels
