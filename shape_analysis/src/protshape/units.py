"""Residue masks and units of analysis: CHAIN_FULL, CHAIN_TRIMMED, DOMAIN, TM_CORE."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .classify import MEMBRANE_CLASSES, domain_label
from .ted import parse_chopping, verify_domain

MIN_SEGMENT = 10          # confident segments shorter than this are excluded
MIN_DOMAIN_RES = 40
MIN_DOMAIN_SSE = 3
TM_PAD = 3
# mask variants for the sensitivity analysis: (pLDDT threshold, require TED domain)
MASK_VARIANTS = {"default": (70.0, True), "plddt50": (50.0, True), "plddt90": (90.0, True), "no_ted": (70.0, False)}


@dataclass
class Unit:
    unit: str
    mask_variant: str
    mask: np.ndarray
    domain_id: str = ""
    meta: dict = field(default_factory=dict)


def dssp(pdb_path: Path, n_res: int) -> np.ndarray:
    """Simplified DSSP codes per residue ('H', 'E', 'C') via mdtraj."""
    import mdtraj as md

    t = md.load_pdb(str(pdb_path))
    codes = md.compute_dssp(t, simplified=True)[0]
    if len(codes) != n_res:
        raise ValueError(f"DSSP returned {len(codes)} residues, structure has {n_res}")
    return np.where(np.isin(codes, ["H", "E"]), codes, "C")


def runs(mask: np.ndarray, resnum: Optional[np.ndarray] = None):
    """(start, end_exclusive) index runs of True, broken also at gaps in residue numbering."""
    out, start = [], None
    for i, m in enumerate(mask):
        brk = resnum is not None and i > 0 and resnum[i] != resnum[i - 1] + 1
        if m and (start is None or brk):
            if start is not None:
                out.append((start, i))
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def sse_segments(ss: np.ndarray, mask: np.ndarray, resnum: np.ndarray) -> list[tuple[str, int]]:
    """Secondary-structure elements inside the masked residues: helices >= 4, strands >= 2 residues."""
    segs = []
    for code, min_len in (("H", 4), ("E", 2)):
        for a, b in runs(mask & (ss == code), resnum):
            if b - a >= min_len:
                segs.append((code, b - a))
    return segs


def drop_short_runs(mask: np.ndarray, resnum: np.ndarray, min_len: int = MIN_SEGMENT) -> np.ndarray:
    out = mask.copy()
    for a, b in runs(mask, resnum):
        if b - a < min_len:
            out[a:b] = False
    return out


def build_units(struct, ann_row, protein_class: str, ted_rows, ss: np.ndarray,
                variants=("default", "plddt50", "plddt90", "no_ted")) -> tuple[list[Unit], dict]:
    """All units for one protein. Returns (units, info) where info records TED verification."""
    n = struct.n_res
    resnum = struct.resnum
    seq = struct.sequence
    info = {"ted_domains": 0, "ted_verified": 0, "ted_shifted": 0, "ted_dropped": 0}
    # verified TED domains in model numbering
    domains = []
    for d in ted_rows:
        info["ted_domains"] += 1
        segs, shift = verify_domain(parse_chopping(d["chopping"]), d.get("md5_domain"), seq)
        if segs is None:
            info["ted_dropped"] += 1
            continue
        info["ted_verified"] += 1
        info["ted_shifted"] += int(shift != 0)
        m = np.zeros(n, bool)
        for a, b in segs:
            m |= (resnum >= a) & (resnum <= b)
        domains.append((d, m, shift))
    in_ted = np.zeros(n, bool)
    for _, m, _ in domains:
        in_ted |= m

    # TM / intramembrane residues
    tm_res = np.zeros(n, bool)
    tm_segments = []
    for a, b in zip(list(ann_row.get("tm_start", [])), list(ann_row.get("tm_end", []))):
        tm_res |= (resnum >= a) & (resnum <= b)
        tm_segments.append((a, b))
    memb_res = tm_res.copy()
    for a, b in zip(list(ann_row.get("intramem_start", [])), list(ann_row.get("intramem_end", []))):
        memb_res |= (resnum >= a) & (resnum <= b)

    units = [Unit("CHAIN_FULL", "default", np.ones(n, bool))]
    keeps = {}
    for v in variants:
        thr, need_ted = MASK_VARIANTS[v]
        keep = struct.plddt >= thr
        if need_ted:
            keep &= in_ted
        keep = drop_short_runs(keep, resnum)
        keeps[v] = keep
        if keep.any():
            units.append(Unit("CHAIN_TRIMMED", v, keep))
        if not need_ted:
            continue                                   # domains are identical to the TED-based variant
        for d, dm, shift in domains:
            m = dm & keep
            if m.sum() < MIN_DOMAIN_RES:
                continue
            sses = sse_segments(ss, m, resnum)
            if len(sses) < MIN_DOMAIN_SSE:
                continue
            n_tm_overlap = sum(int(((resnum >= a) & (resnum <= b) & m).sum() >= 5) for a, b in tm_segments)
            frac_tm = float((memb_res & m).sum() / m.sum())
            cath = d.get("cath_label") or ""
            meta = {
                "ted_id": d["ted_id"], "ted_chopping": d["chopping"], "ted_shift": shift,
                "ted_consensus": d.get("consensus_level"), "ted_plddt": d.get("plddt"),
                "ted_packing_density": d.get("packing_density"), "ted_norm_rg": d.get("norm_rg"),
                "cath_label": cath, "cath_class": int(cath.split(".")[0]) if cath[:1].isdigit() else np.nan,
                "domain_n_tm_overlap": n_tm_overlap, "domain_frac_tm": frac_tm,
                "domain_label": domain_label(protein_class, n_tm_overlap, frac_tm),
            }
            units.append(Unit("DOMAIN", v, m, domain_id=d["ted_id"].split("_")[-1], meta=meta))
    if protein_class in MEMBRANE_CLASSES and tm_segments:
        core = np.zeros(n, bool)
        for a, b in tm_segments:
            core |= (resnum >= a - TM_PAD) & (resnum <= b + TM_PAD)
        units.append(Unit("TM_CORE", "default", core))
    info["frac_excluded"] = float(1 - keeps.get("default", np.zeros(n, bool)).mean()) if n else np.nan
    info["in_ted_frac"] = float(in_ted.mean()) if n else np.nan
    return units, info
