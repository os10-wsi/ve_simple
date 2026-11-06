"""Validate (and optionally delete) per-protein cache files so that broken ones are recomputed.

Concurrently running duplicate jobs, killed jobs or full disks can leave truncated or garbled
files in `cache/<accession>/`.  Every file type the pipeline caches is checked here with the same
rules the pipeline applies when it reads them.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .constants import AMINO_ACIDS

log = logging.getLogger(__name__)

SCORE_TABLES = {  # cache file -> required columns
    "scores_esm1v.tsv": ["position", "wt", "mut", "esm1v"],
    "scores_saprot.tsv": ["position", "wt", "mut", "saprot"],
    "scores_thermompnn.tsv": ["position", "wt", "mut", "thermompnn"],
    "scores_foldx.tsv": ["position", "wt", "mut", "foldx"],
    "scores_exposure.tsv": ["position", "wt", "solvent_exposure", "plddt"],
}


def n_standard(sequence: str) -> int:
    return sum(aa in AMINO_ACIDS for aa in sequence)


def check_score_table(path: Path, sequence: Optional[str] = None) -> Optional[str]:
    """Return a problem description, or None if the table is usable."""
    cols = SCORE_TABLES.get(path.name)
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as e:  # noqa: BLE001
        return f"unreadable ({e.__class__.__name__})"
    if cols and any(c not in df.columns for c in cols):
        return f"missing columns {sorted(set(cols) - set(df.columns))}"
    if len(df) == 0:
        return "empty"
    if path.name != "scores_exposure.tsv":
        if len(df) % 19 != 0:
            return f"{len(df)} rows is not a multiple of 19 (truncated?)"
        if path.name == "scores_esm1v.tsv" and sequence and len(df) != 19 * n_standard(sequence):
            return f"{len(df)} rows, expected {19 * n_standard(sequence)}"
        score_col = cols[-1]
        if not pd.api.types.is_numeric_dtype(df[score_col]):
            return f"non-numeric {score_col} column"
    if sequence is not None and len(df):
        pos = df["position"].to_numpy()
        if pos.max() > len(sequence) or pos.min() < 1:
            return "positions outside the sequence"
    return None


def check_npy(path: Path, sequence: Optional[str]) -> Optional[str]:
    try:
        mat = np.load(path)
    except Exception as e:  # noqa: BLE001
        return f"unreadable ({e.__class__.__name__})"
    if mat.ndim != 2 or mat.shape[1] != len(AMINO_ACIDS):
        return f"bad shape {mat.shape}"
    if sequence is not None and mat.shape[0] != len(sequence):
        return f"{mat.shape[0]} rows, expected {len(sequence)}"
    return None


def check_json(path: Path) -> Optional[str]:
    try:
        json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        return f"unreadable ({e.__class__.__name__})"
    return None


def check_pdb(path: Path) -> Optional[str]:
    try:
        text = path.read_text()
    except Exception as e:  # noqa: BLE001
        return f"unreadable ({e.__class__.__name__})"
    if "ATOM" not in text:
        return "no ATOM records"
    if not text.rstrip().endswith("END"):
        return "no END record (truncated?)"
    return None


def check_accession_dir(acc_dir: Path) -> list[tuple[Path, str]]:
    """All problems found under one accession directory."""
    problems: list[tuple[Path, str]] = []
    sequence = None
    up = acc_dir / "uniprot.json"
    if up.exists():
        prob = check_json(up)
        if prob:
            problems.append((up, prob))
        else:
            try:
                sequence = json.loads(up.read_text())["sequence"]["value"]
            except Exception:  # noqa: BLE001
                problems.append((up, "no sequence"))
    for p in acc_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if name.endswith(".tmp") or ".tmp." in name or name.endswith(".part"):
            problems.append((p, "leftover temporary file"))
        elif name in SCORE_TABLES:
            prob = check_score_table(p, sequence)
            if prob:
                problems.append((p, prob))
        elif name.startswith("esm1v_") and name.endswith(".npy"):
            prob = check_npy(p, sequence)
            if prob:
                problems.append((p, prob))
        elif name.endswith(".json") and name != "uniprot.json":
            prob = check_json(p)
            if prob:
                problems.append((p, prob))
        elif name.endswith(".pdb"):
            prob = check_pdb(p)
            if prob:
                problems.append((p, prob))
        elif name.endswith("-aa-substitutions.csv") and p.stat().st_size > 0:
            try:
                pd.read_csv(p, usecols=["protein_variant", "am_pathogenicity"])
            except Exception as e:  # noqa: BLE001
                problems.append((p, f"unreadable ({e.__class__.__name__})"))
    return problems


def check_cache(cache_dir: Path, delete: bool = False, accessions: Optional[list[str]] = None) -> list[tuple[Path, str]]:
    cache_dir = Path(cache_dir)
    dirs = [cache_dir / a for a in accessions] if accessions else sorted(
        d for d in cache_dir.iterdir() if d.is_dir() and not d.name.startswith(("_", ".")) and d.name != "huggingface")
    all_problems: list[tuple[Path, str]] = []
    for d in dirs:
        if not d.exists():
            continue
        probs = check_accession_dir(d)
        for path, why in probs:
            log.warning("%s: %s", path, why)
            if delete:
                path.unlink()
                # a broken final ESM table must also drop the per-model matrices' claim to completeness
                if path.name == "scores_esm1v.tsv":
                    pass
        all_problems.extend(probs)
    return all_problems
