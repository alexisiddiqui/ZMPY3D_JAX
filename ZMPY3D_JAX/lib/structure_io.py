"""Biotite-backed PDB/mmCIF loading for new all-atom workflows."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
from biotite.structure.io import pdb, pdbx

import ZMPY3D_JAX.config as _config


class AtomicStructure(NamedTuple):
    """The metadata and arrays consumed by atomic voxelization."""

    id: str
    xyz: np.ndarray
    elements: tuple[str, ...]
    occupancies: np.ndarray


_WATER_NAMES = {"DOD", "HOH", "WAT"}


def load_atomic_structures(
    file_name: str,
    *,
    model: int | None = 1,
    chain_ids: Sequence[str] | str | None = None,
    assembly_id: str | None = None,
    include_hetero: bool = False,
    include_water: bool = False,
    include_hydrogens: bool = False,
) -> list[AtomicStructure]:
    """Load one model, or every model when ``model=None``, from PDB/mmCIF."""
    if model is not None and model < 1:
        raise ValueError("model must be a positive one-based index or None")
    chains = (
        None
        if chain_ids is None
        else {chain_ids}
        if isinstance(chain_ids, str)
        else set(chain_ids)
    )
    suffix = Path(file_name).suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        source = pdbx.CIFFile.read(file_name)
        block = source[next(iter(source.keys()))]
        model_count = len(
            np.unique(block["atom_site"]["pdbx_PDB_model_num"].as_array(np.int32))
        )
        getter = pdbx.get_structure if assembly_id is None else pdbx.get_assembly
    else:
        if assembly_id is not None:
            raise ValueError("assembly_id is supported only for mmCIF input")
        source = pdb.PDBFile.read(file_name)
        model_count = source.get_model_count()
        getter = pdb.get_structure
    models = [model] if model is not None else range(1, model_count + 1)
    samples: list[AtomicStructure] = []
    for model_index in models:
        kwargs = {
            "model": model_index,
            "altloc": "occupancy",
            "extra_fields": ["occupancy"],
        }
        if assembly_id is not None:
            kwargs["assembly_id"] = assembly_id
        atoms = getter(source, **kwargs)

        keep = np.ones(atoms.array_length(), dtype=bool)
        if chains is not None:
            keep &= np.isin(atoms.chain_id, tuple(chains))
        if not include_hetero:
            keep &= ~atoms.hetero
        if not include_water:
            keep &= ~np.isin(np.char.upper(atoms.res_name), tuple(_WATER_NAMES))
        elements = np.char.upper(atoms.element.astype(str))
        if not include_hydrogens:
            keep &= ~np.isin(elements, ("H", "D"))
        atoms = atoms[keep]
        elements = elements[keep]
        elements[elements == "D"] = "H"
        occupancies = np.asarray(atoms.occupancy, dtype=_config.FLOAT_DTYPE)
        if np.any(~np.isfinite(occupancies)) or np.any(occupancies < 0):
            raise ValueError(f"Invalid occupancy in {file_name}, model {model_index}")

        parts = [str(Path(file_name)), f"model={model_index}"]
        if assembly_id is not None:
            parts.append(f"assembly={assembly_id}")
        if chains is not None:
            parts.append(f"chains={','.join(sorted(chains))}")
        samples.append(
            AtomicStructure(
                id=":".join(parts),
                xyz=np.asarray(atoms.coord, dtype=_config.FLOAT_DTYPE).reshape((-1, 3)),
                elements=tuple(elements.tolist()),
                occupancies=occupancies,
            )
        )
    return samples
