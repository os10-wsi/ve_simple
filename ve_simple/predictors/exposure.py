"""Per-residue solvent exposure (relative SASA) and pLDDT from an AlphaFold model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..alphafold import read_pdb_chain
from ..constants import MAX_ASA, THREE_TO_ONE


def solvent_exposure(pdb_path: Path) -> pd.DataFrame:
    """Shrake-Rupley SASA per residue, normalised by the Tien et al. maximum ASA.

    Returns columns: position (PDB residue number), wt, sasa, solvent_exposure (RSA), plddt.
    """
    from Bio.PDB.SASA import ShrakeRupley

    chain, seq, resnums, plddt = read_pdb_chain(pdb_path)
    sr = ShrakeRupley()
    sr.compute(chain, level="R")
    rows = []
    for res, aa, num, pl in zip([r for r in chain if r.id[0] == " "], seq, resnums, plddt):
        sasa = float(getattr(res, "sasa", np.nan))
        max_asa = MAX_ASA.get(aa)
        rsa = sasa / max_asa if max_asa else np.nan
        rows.append({"position": int(num), "wt": aa, "sasa": sasa,
                     "solvent_exposure": min(rsa, 1.5) if np.isfinite(rsa) else np.nan,
                     "plddt": float(pl)})
    return pd.DataFrame(rows)
