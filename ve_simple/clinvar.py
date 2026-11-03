"""ClinVar missense classifications via the EBI Proteins API (variation endpoint).

The Proteins API aggregates ClinVar, UniProt and Ensembl variants per UniProt accession and
maps them onto protein coordinates, so no genome-to-protein lift-over is needed.  Only
classifications whose source list includes ClinVar are used.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .constants import AMINO_ACIDS
from .utils import http_get, read_json, write_json

log = logging.getLogger(__name__)

VARIATION_API = "https://www.ebi.ac.uk/proteins/api/variation/{acc}"

# ClinVar significance strings -> coarse class
_CLASS_MAP = {
    "pathogenic": "pathogenic",
    "likely pathogenic": "likely_pathogenic",
    "pathogenic/likely pathogenic": "likely_pathogenic",
    "benign": "benign",
    "likely benign": "likely_benign",
    "benign/likely benign": "likely_benign",
    "variant of uncertain significance": "vus",
    "uncertain significance": "vus",
    "conflicting interpretations of pathogenicity": "conflicting",
    "conflicting classifications of pathogenicity": "conflicting",
}

CLINVAR_COLUMNS = ["accession", "position", "wt", "mut", "clinvar_class", "clinvar_label",
                   "clinvar_significances", "clinvar_review_status", "clinvar_ids"]


def fetch_variation(acc: str, cache_dir: Path) -> Optional[dict]:
    cache = Path(cache_dir) / acc / "ebi_variation.json"
    if cache.exists():
        return read_json(cache)
    r = http_get(VARIATION_API.format(acc=acc), headers={"Accept": "application/json"}, timeout=300)
    if r.status_code == 404:
        write_json({}, cache)
        return None
    r.raise_for_status()
    d = r.json()
    write_json(d, cache)
    return d


def _coarse_class(sig: str) -> str:
    return _CLASS_MAP.get(sig.strip().lower(), "other")


def _resolve(classes: set[str]) -> str:
    """Collapse the set of ClinVar classes reported for one variant into a single class."""
    path = {"pathogenic", "likely_pathogenic"} & classes
    ben = {"benign", "likely_benign"} & classes
    if path and ben:
        return "conflicting"
    if "conflicting" in classes and not (path or ben):
        return "conflicting"
    if path:
        return "pathogenic" if "pathogenic" in classes else "likely_pathogenic"
    if ben:
        return "benign" if "benign" in classes else "likely_benign"
    if "vus" in classes:
        return "vus"
    return "other"


def parse_clinvar_missense(acc: str, data: Optional[dict]) -> pd.DataFrame:
    """Extract ClinVar-classified single amino-acid substitutions.

    Returns one row per (position, wt, mut) with the resolved class.
    """
    rows: dict[tuple, dict] = {}
    if not data:
        return pd.DataFrame(columns=CLINVAR_COLUMNS)
    for f in data.get("features", []):
        if f.get("type") != "VARIANT":
            continue
        if f.get("consequenceType") != "missense":
            continue
        wt, mut = f.get("wildType"), f.get("mutatedType")
        if not wt or not mut or len(wt) != 1 or len(mut) != 1:
            continue
        if wt not in AMINO_ACIDS or mut not in AMINO_ACIDS or wt == mut:
            continue
        if f.get("begin") != f.get("end"):
            continue
        try:
            pos = int(f["begin"])
        except (KeyError, ValueError):
            continue
        sigs = [cs for cs in (f.get("clinicalSignificances") or []) if "ClinVar" in (cs.get("sources") or [])]
        if not sigs:
            continue
        ids = sorted({x["id"] for x in f.get("xrefs", []) if x.get("name") == "ClinVar"})
        key = (pos, wt, mut)
        row = rows.setdefault(key, {"sigs": set(), "review": set(), "ids": set()})
        for cs in sigs:
            row["sigs"].add(cs.get("type", ""))
            if cs.get("reviewStatus"):
                row["review"].add(cs["reviewStatus"])
        row["ids"].update(ids)
    out = []
    for (pos, wt, mut), row in sorted(rows.items()):
        classes = {_coarse_class(s) for s in row["sigs"]}
        cls = _resolve(classes)
        out.append({
            "accession": acc, "position": pos, "wt": wt, "mut": mut,
            "clinvar_class": cls,
            "clinvar_label": class_to_label(cls),
            "clinvar_significances": "|".join(sorted(row["sigs"])),
            "clinvar_review_status": "|".join(sorted(row["review"])),
            "clinvar_ids": "|".join(sorted(row["ids"])),
        })
    return pd.DataFrame(out, columns=CLINVAR_COLUMNS)


def class_to_label(cls: str, include_likely: bool = True) -> str:
    """Map the fine class to the three bins used for benchmarking: pathogenic / benign / vus."""
    if cls == "pathogenic" or (include_likely and cls == "likely_pathogenic"):
        return "pathogenic"
    if cls == "benign" or (include_likely and cls == "likely_benign"):
        return "benign"
    if cls == "vus":
        return "vus"
    return "other"


def summarise(df: pd.DataFrame) -> dict:
    c = df["clinvar_label"].value_counts() if len(df) else {}
    fine = df["clinvar_class"].value_counts() if len(df) else {}
    return {
        "n_clinvar_missense": int(len(df)),
        "n_pathogenic": int(c.get("pathogenic", 0)),
        "n_benign": int(c.get("benign", 0)),
        "n_vus": int(c.get("vus", 0)),
        "n_other": int(c.get("other", 0)),
        "n_pathogenic_strict": int(fine.get("pathogenic", 0)),
        "n_likely_pathogenic": int(fine.get("likely_pathogenic", 0)),
        "n_benign_strict": int(fine.get("benign", 0)),
        "n_likely_benign": int(fine.get("likely_benign", 0)),
        "n_conflicting": int(fine.get("conflicting", 0)),
    }
