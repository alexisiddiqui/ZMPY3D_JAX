import numpy as np
import pytest

import ZMPY3D_JAX as z


def _atom_line(serial, atom, altloc, residue, x, y, z_coord):
    return (
        f"ATOM  {serial:5d} {atom:^4s}{altloc:1s}{residue:>3s} A   1    "
        f"{x:8.3f}{y:8.3f}{z_coord:8.3f}  1.00 20.00           C\n"
    )


def test_parser_selects_ca_and_primary_altloc(tmp_path):
    pdb = tmp_path / "trace.pdb"
    pdb.write_text(
        _atom_line(1, "N", " ", "ALA", 0, 0, 0)
        + _atom_line(2, "CA", "A", "ALA", 1, 2, 3)
        + _atom_line(3, "CA", "B", "ALA", 9, 9, 9),
        encoding="utf-8",
    )
    xyz, residues = z.get_pdb_xyz_ca(str(pdb))
    np.testing.assert_allclose(xyz, [[1, 2, 3]])
    assert residues == ["ALA"]


def test_empty_parser_result_has_coordinate_shape(tmp_path):
    pdb = tmp_path / "empty.pdb"
    pdb.write_text("END\n", encoding="utf-8")
    xyz, residues = z.get_pdb_xyz_ca(str(pdb))
    assert xyz.shape == (0, 3)
    assert residues == []


def test_malformed_ca_coordinates_raise_clear_error(tmp_path):
    pdb = tmp_path / "bad.pdb"
    line = _atom_line(1, "CA", " ", "ALA", 1, 2, 3)
    pdb.write_text(line[:30] + "not-a-x " + line[38:], encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed CA coordinates at line 1"):
        z.get_pdb_xyz_ca(str(pdb))
