import urllib.request
from pathlib import Path

import pytest


@pytest.fixture
def pdb_files(tmp_path):
    """
    Fixture to download PDB files for testing.

    Returns a dictionary with PDB file paths.
    """
    pdb_urls = {
        "6NT5": "https://github.com/tawssie/ZMPY3D/raw/main/6NT5.pdb",
        "6NT6": "https://github.com/tawssie/ZMPY3D/raw/main/6NT6.pdb",
    }

    pdb_paths = {}
    for name, url in pdb_urls.items():
        file_path = tmp_path / f"{name}.pdb"
        urllib.request.urlretrieve(url, file_path)
        pdb_paths[name] = str(file_path)

    return pdb_paths


@pytest.fixture
def output_dir():
    """
    Fixture to create and return the output directory path.
    """
    output_path = Path(__file__).parent.parent / "integration" / "_super_output"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
