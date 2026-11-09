"""Map residue numbers of externally sourced tables onto the current UniProt sequence.

Predictions taken from AlphaFold models or from the Zenodo ThermoMPNN archive were computed on the
sequence UniProt had at the time (2021-2022).  When UniProt has since changed the canonical
sequence, residue numbers no longer line up.  Tables are therefore checked against the current
sequence and, if they differ, re-numbered through a global alignment; only residues that align to an
identical residue keep their value, so nothing is attached to the wrong residue.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MAX_ALIGN_CELLS = 4e8   # ~20k x 20k residues; larger pairs are not aligned (values dropped instead)


def table_sequence(df: pd.DataFrame) -> str:
    """Sequence implied by a table's (position, wt) columns; unknown positions are 'X'."""
    wt = df.drop_duplicates("position").set_index("position")["wt"]
    n = int(wt.index.max()) if len(wt) else 0
    seq = ["X"] * n
    for p, a in wt.items():
        if 1 <= p <= n:
            seq[int(p) - 1] = str(a)
    return "".join(seq)


def matches_sequence(df: pd.DataFrame, target: str) -> bool:
    pos = df["position"].to_numpy()
    if len(pos) == 0:
        return True
    if pos.min() < 1 or pos.max() > len(target):
        return False
    tgt = np.frombuffer(target.encode(), dtype="S1")
    return bool((tgt[pos - 1] == df["wt"].to_numpy().astype("S1")).all())


def position_map(source: str, target: str) -> dict[int, int]:
    """1-based source position -> 1-based target position, for identical aligned residues only."""
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                              open_gap_score=-5, extend_gap_score=-0.5)
    aln = aligner.align(source, target)[0]
    out: dict[int, int] = {}
    for (s0, s1), (t0, _t1) in zip(*aln.aligned):
        for k in range(s1 - s0):
            if source[s0 + k] == target[t0 + k] and source[s0 + k] != "X":
                out[s0 + k + 1] = t0 + k + 1
    return out


def remap_to_sequence(df: Optional[pd.DataFrame], target: str, label: str = "") -> tuple[Optional[pd.DataFrame], Optional[float]]:
    """Return (table in target numbering, fraction of source residues kept) — fraction is None if the
    table already matched the target and was left untouched."""
    if df is None or len(df) == 0 or matches_sequence(df, target):
        return df, None
    source = table_sequence(df)
    if len(source) * len(target) > MAX_ALIGN_CELLS:
        log.warning("%s: sequence differs from UniProt and is too long to align; values dropped", label)
        return df.iloc[0:0], 0.0
    pmap = position_map(source, target)
    out = df[df["position"].isin(pmap.keys())].copy()
    out["position"] = out["position"].map(pmap).astype(int)
    n_src = len(set(df["position"]))
    frac = len(pmap) / n_src if n_src else 0.0
    log.info("%s: sequence differs from current UniProt sequence; %d of %d residues mapped by alignment",
             label, len(pmap), n_src)
    return out, frac
