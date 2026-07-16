from pathlib import Path

import pytest


@pytest.fixture
def pdb_files():
    """Return committed PDB fixtures without requiring network access."""
    repo_root = Path(__file__).resolve().parents[3]
    paths = {name: repo_root / f"{name}.pdb" for name in ("6NT5", "6NT6")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        pytest.fail(f"Missing committed PDB fixture(s): {', '.join(missing)}")
    return {name: str(path) for name, path in paths.items()}


@pytest.fixture
def output_dir(tmp_path):
    """Create integration artifacts outside the source tree."""
    output_path = tmp_path / "super_output"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
