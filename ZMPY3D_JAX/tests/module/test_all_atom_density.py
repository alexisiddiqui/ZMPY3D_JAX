import numpy as np
import pytest
from biotite.structure import AtomArray
from biotite.structure.io import pdbx

import ZMPY3D_JAX as z
from ZMPY3D_JAX.lib.fill_voxel_by_weight_density04 import (
    fill_voxel_by_weight_density_host,
)
from ZMPY3D_JAX.ZMPY3D_CLI_BatchZM import _chunk_by_voxel_budget


def _atom_line(
    record,
    serial,
    atom,
    altloc,
    residue,
    chain,
    residue_number,
    x,
    y,
    z_coord,
    occupancy,
    element,
):
    return (
        f"{record:<6s}{serial:5d} {atom:^4s}{altloc:1s}{residue:>3s} "
        f"{chain:1s}{residue_number:4d}    {x:8.3f}{y:8.3f}{z_coord:8.3f}"
        f"{occupancy:6.2f}{20.0:6.2f}          {element:>2s}\n"
    )


def test_all_atom_parser_filters_and_resolves_altloc(tmp_path):
    pdb = tmp_path / "atoms.pdb"
    pdb.write_text(
        _atom_line("ATOM", 1, "N", "", "ALA", "A", 1, 0, 0, 0, 1.0, "N")
        + _atom_line("ATOM", 2, "CA", "A", "ALA", "A", 1, 1, 0, 0, 0.4, "C")
        + _atom_line("ATOM", 3, "CA", "B", "ALA", "A", 1, 2, 0, 0, 0.8, "C")
        + _atom_line("ATOM", 4, "H", "", "ALA", "A", 1, 0, 1, 0, 1.0, "H")
        + _atom_line("HETATM", 5, "S", "", "SO4", "A", 2, 3, 0, 0, 0.5, "S")
        + _atom_line("HETATM", 6, "O", "", "HOH", "A", 3, 4, 0, 0, 1.0, "O"),
        encoding="utf-8",
    )

    default = z.load_atomic_structures(str(pdb))[0]
    assert default.elements == ("N", "C")
    np.testing.assert_allclose(default.xyz, [[0, 0, 0], [2, 0, 0]])
    np.testing.assert_allclose(default.occupancies, [1.0, 0.8])

    expanded = z.load_atomic_structures(
        str(pdb), include_hetero=True, include_hydrogens=True
    )[0]
    assert expanded.elements == ("N", "C", "H", "S")


def test_all_atom_parser_selects_chain_and_model(tmp_path):
    pdb = tmp_path / "models.pdb"
    pdb.write_text(
        "MODEL        1\n"
        + _atom_line("ATOM", 1, "C", "", "GLY", "A", 1, 1, 0, 0, 1.0, "C")
        + _atom_line("ATOM", 2, "C", "", "GLY", "B", 1, 2, 0, 0, 1.0, "C")
        + "ENDMDL\nMODEL        2\n"
        + _atom_line("ATOM", 3, "C", "", "GLY", "A", 1, 3, 0, 0, 1.0, "C")
        + "ENDMDL\n",
        encoding="utf-8",
    )

    atoms = z.load_atomic_structures(str(pdb), chain_ids="A", model=2)[0]
    np.testing.assert_allclose(atoms.xyz, [[3, 0, 0]])
    ensemble = z.load_atomic_structures(str(pdb), chain_ids="A", model=None)
    assert len(ensemble) == 2
    assert ensemble[0].id.endswith("model=1:chains=A")
    assert ensemble[1].id.endswith("model=2:chains=A")


def test_mmcif_loader_uses_same_atomic_structure(tmp_path):
    atoms = AtomArray(2)
    atoms.coord = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = 1
    atoms.res_name[:] = "GLY"
    atoms.atom_name[:] = ["N", "CA"]
    atoms.element[:] = ["N", "C"]
    atoms.hetero[:] = False
    atoms.set_annotation("occupancy", np.array([1.0, 0.5]))
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, atoms)
    path = tmp_path / "atoms.cif"
    cif.write(path)

    loaded = z.load_atomic_structures(str(path))[0]
    assert loaded.elements == ("N", "C")
    np.testing.assert_allclose(loaded.xyz, atoms.coord)
    np.testing.assert_allclose(loaded.occupancies, [1.0, 0.5])


def test_atomic_gaussians_preserve_integrated_mass_and_occupancy():
    grid_width = 0.5
    boxes = z.get_atomic_gaussian_density_cache(grid_width)
    masses = z.get_atomic_mass_map()
    for element in ("C", "N", "O", "S"):
        integrated_mass = np.sum(boxes[element]) * grid_width**3 / 100.0
        assert integrated_mass == pytest.approx(masses[element], rel=2e-6)

    full, _ = fill_voxel_by_weight_density_host(
        np.array([[0.0, 0.0, 0.0]]),
        ["C"],
        masses,
        grid_width,
        boxes,
        point_weight_multipliers=[1.0],
    )
    quarter, _ = fill_voxel_by_weight_density_host(
        np.array([[0.0, 0.0, 0.0]]),
        ["C"],
        masses,
        grid_width,
        boxes,
        point_weight_multipliers=[0.25],
    )
    assert np.sum(quarter) == pytest.approx(np.sum(full) * 0.25, rel=2e-6)


def test_all_atom_descriptor_is_explicit_and_device_native(tmp_path):
    pdb = tmp_path / "asymmetric.pdb"
    pdb.write_text(
        _atom_line("ATOM", 1, "N", "", "ALA", "A", 1, 0, 0, 0, 1.0, "N")
        + _atom_line("ATOM", 2, "CA", "", "ALA", "A", 1, 2, 0, 0, 1.0, "C")
        + _atom_line("ATOM", 3, "C", "", "ALA", "A", 1, 2, 1, 0, 1.0, "C")
        + _atom_line("ATOM", 4, "O", "", "ALA", "A", 1, 2, 1, 1, 1.0, "O"),
        encoding="utf-8",
    )

    descriptor = z.ZMPY3D_CLI_ZM(
        str(pdb), Mode=1, Representation="all_atom_gaussian"
    )
    assert isinstance(descriptor, z.DescriptorVector)
    assert descriptor.values.shape == (16,)
    assert bool(np.all(np.asarray(descriptor.is_valid)))


def test_descriptor_representation_must_be_known(tmp_path):
    with pytest.raises(ValueError, match="Representation must be"):
        z.ZMPY3D_CLI_ZM(
            str(tmp_path / "unused.pdb"), Mode=1, Representation="unknown"
        )


def test_voxel_chunks_sort_and_round_without_exceeding_budget():
    items = [
        (0, "large", np.ones((17, 9, 9))),
        (1, "small", np.ones((3, 3, 3))),
        (2, "medium", np.ones((9, 7, 7))),
    ]
    chunks = _chunk_by_voxel_budget(items, batch_size=2, voxel_budget=20_000)
    flattened_ids = [item[1] for chunk, _ in chunks for item in chunk]
    assert flattened_ids == ["small", "medium", "large"]
    for chunk, shape in chunks:
        assert all(size % 8 == 0 for size in shape)
        assert len(chunk) * np.prod(shape) <= 20_000


def test_heterogeneous_batch_returns_ids_representation_and_failures(tmp_path):
    first = tmp_path / "first.pdb"
    second = tmp_path / "second.pdb"
    first.write_text(
        _atom_line("ATOM", 1, "N", "", "ALA", "A", 1, 0, 0, 0, 1.0, "N")
        + _atom_line("ATOM", 2, "CA", "", "ALA", "A", 1, 2, 0, 0, 1.0, "C")
        + _atom_line("ATOM", 3, "O", "", "ALA", "A", 1, 2, 1, 1, 1.0, "O"),
        encoding="utf-8",
    )
    second.write_text(
        first.read_text(encoding="utf-8")
        + _atom_line("ATOM", 4, "S", "", "CYS", "A", 2, 8, 2, 1, 1.0, "S"),
        encoding="utf-8",
    )

    result = z.calculate_structure_descriptors_batch(
        [str(first), str(tmp_path / "missing.pdb"), str(second)],
        representation="all_atom_gaussian",
        mode=1,
        batch_size=2,
        on_error="skip",
    )
    assert result.ids == (
        f"{first}:model=1",
        f"{second}:model=1",
    )
    assert result.representation == "all_atom_gaussian"
    assert result.values.shape == (2, 16)
    assert bool(np.all(np.asarray(result.is_valid)))
    assert len(result.failures) == 1
    assert result.failures[0].id.endswith("missing.pdb")
