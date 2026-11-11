"""Atomic radii, masses and hydrophobicity scales."""
from __future__ import annotations

import numpy as np

# Bondi van der Waals radii (A) for the heavy atoms found in AlphaFold models.
VDW_RADIUS = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "SE": 1.90}
ATOMIC_MASS = {"C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06, "SE": 78.97}
PROBE_RADIUS = 1.4

# Kyte & Doolittle (1982) hydropathy; higher = more hydrophobic.
KYTE_DOOLITTLE = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5, "GLN": -3.5, "GLU": -3.5,
    "GLY": -0.4, "HIS": -3.2, "ILE": 4.5, "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8,
    "PRO": -1.6, "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}
# Wimley & White (1996) whole-residue water->octanol transfer free energies (kcal/mol; Asp, Glu,
# His charged). Stored NEGATED so that, like Kyte-Doolittle, higher = more hydrophobic.
_WW_OCTANOL_DG = {
    "ALA": 0.50, "ARG": 1.81, "ASN": 0.85, "ASP": 3.64, "CYS": -0.02, "GLN": 0.77, "GLU": 3.63,
    "GLY": 1.15, "HIS": 2.33, "ILE": -1.12, "LEU": -1.25, "LYS": 2.80, "MET": -0.67, "PHE": -1.71,
    "PRO": 0.14, "SER": 0.46, "THR": 0.25, "TRP": -2.09, "TYR": -0.71, "VAL": -0.46,
}
WIMLEY_WHITE_OCT = {k: -v for k, v in _WW_OCTANOL_DG.items()}

# Carbons bonded to O/N by a double or partial-double bond (carbonyl, carboxyl, amide, guanidino).
# They are not aliphatic/aromatic and score 0 on the atomic scale.
_POLAR_CARBONS = {("*", "C"), ("ASP", "CG"), ("GLU", "CD"), ("ASN", "CG"), ("GLN", "CD"), ("ARG", "CZ")}


def atomic_hydrophobicity(resname: str, atom_name: str, element: str) -> float:
    """Atomic scale: aliphatic/aromatic C = +1, N/O = -1, S = 0 (carbonyl-type C = 0)."""
    if element == "C":
        if (resname, atom_name) in _POLAR_CARBONS or ("*", atom_name) in _POLAR_CARBONS:
            return 0.0
        return 1.0
    if element in ("N", "O"):
        return -1.0
    return 0.0


SCALES = ("kd", "ww", "atom")


def atom_scale_values(resnames: np.ndarray, atom_names: np.ndarray, elements: np.ndarray) -> dict[str, np.ndarray]:
    """Per-atom hydrophobicity for each scale (residue scales broadcast to the residue's atoms)."""
    kd = np.array([KYTE_DOOLITTLE.get(r, 0.0) for r in resnames])
    ww = np.array([WIMLEY_WHITE_OCT.get(r, 0.0) for r in resnames])
    at = np.array([atomic_hydrophobicity(r, a, e) for r, a, e in zip(resnames, atom_names, elements)])
    return {"kd": kd, "ww": ww, "atom": at}
