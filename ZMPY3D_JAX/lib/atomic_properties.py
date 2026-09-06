"""Atomic properties used by the optional all-atom density representation."""

from typing import Dict


# Bondi-style elemental van der Waals radii in Angstroms.  The initial table is
# deliberately limited to elements for which this representation has a clear,
# documented meaning.  Unsupported elements fail loudly rather than receiving
# an arbitrary fallback radius.
BONDI_VDW_RADII: Dict[str, float] = {
    "H": 1.20,
    "B": 1.92,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "SI": 2.10,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "SE": 1.90,
    "BR": 1.85,
    "I": 1.98,
}


STANDARD_ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.008,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403,
    "SI": 28.085,
    "P": 30.973762,
    "S": 32.06,
    "CL": 35.45,
    "SE": 78.971,
    "BR": 79.904,
    "I": 126.90447,
}


def get_bondi_vdw_radius_map() -> Dict[str, float]:
    """Return a copy of the supported Bondi-style vdW radius table."""
    return dict(BONDI_VDW_RADII)


def get_atomic_mass_map() -> Dict[str, float]:
    """Return a copy of the standard atomic-mass table used for weighting."""
    return dict(STANDARD_ATOMIC_MASSES)
