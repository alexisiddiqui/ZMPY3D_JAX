"""Report occupied-grid and scaled-radius statistics for committed PDB inputs."""
from pathlib import Path
import json
import numpy as np

import ZMPY3D_JAX as z
from ZMPY3D_JAX.ZMPY3D_CLI_BatchZM import _prepare_batch_runtime
from ZMPY3D_JAX.lib.fill_voxel_by_weight_density04 import fill_voxel_by_weight_density_host


def main():
    root = Path(__file__).resolve().parents[3]
    runtime = _prepare_batch_runtime(1.0, 6)
    rows = []
    for path in [root / "6NT5.pdb", root / "6NT6.pdb", root / "ZMPY3D_JAX/tests/data/9j1r_ca_regression.pdb"]:
        xyz, residues = z.get_pdb_xyz_ca(str(path))
        voxel, _ = fill_voxel_by_weight_density_host(xyz, residues, runtime.param["residue_weight_map"], 1.0, runtime.residue_box[1.0])
        occupied = np.argwhere(voxel != 0)
        rows.append({"structure": path.name, "shape": list(voxel.shape), "occupancy_fraction": float(len(occupied)/voxel.size)})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
