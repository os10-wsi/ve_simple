"""Sequence-only physico-chemical predictors: BLOSUM62, delta volume/charge/hydrophobicity."""
from __future__ import annotations

import pandas as pd

from ..constants import AMINO_ACIDS, BLOSUM62, CHARGE, HYDROPHOBICITY, VOLUME


def saturation_table(sequence: str) -> pd.DataFrame:
    """All 19 substitutions at every position of `sequence` (1-based positions)."""
    rows = []
    for i, wt in enumerate(sequence, start=1):
        if wt not in AMINO_ACIDS:
            continue
        for mut in AMINO_ACIDS:
            if mut != wt:
                rows.append((i, wt, mut))
    return pd.DataFrame(rows, columns=["position", "wt", "mut"])


def physchem_scores(variants: pd.DataFrame) -> pd.DataFrame:
    """Add blosum62 / delta_* / abs_delta_* columns to a (position, wt, mut) table."""
    out = variants.copy()
    wt, mut = out["wt"], out["mut"]
    out["blosum62"] = [BLOSUM62.get((a, b)) for a, b in zip(wt, mut)]
    out["delta_volume"] = [VOLUME[b] - VOLUME[a] for a, b in zip(wt, mut)]
    out["delta_charge"] = [CHARGE[b] - CHARGE[a] for a, b in zip(wt, mut)]
    out["delta_hydrophobicity"] = [HYDROPHOBICITY[b] - HYDROPHOBICITY[a] for a, b in zip(wt, mut)]
    for c in ("delta_volume", "delta_charge", "delta_hydrophobicity"):
        out["abs_" + c] = out[c].abs()
    return out
