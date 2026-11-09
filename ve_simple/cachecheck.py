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


def expected_esm1v_rows(sequence: str) -> int:
    """ESM-1v tables list every alternative amino acid at every position: 19 at a standard
    residue, 20 at a non-standard one (e.g. selenocysteine 'U' scored as 'X')."""
    return sum(19 if aa in AMINO_ACIDS else 20 for aa in sequence)


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
    if cols:
        key = [c for c in cols if c in ("position", "wt", "mut")]
        if df[key].isna().any().any():
            return "incomplete rows (truncated?)"
        score_col = cols[-1]
        if not pd.api.types.is_numeric_dtype(df[score_col]):
            return f"non-numeric {score_col} column"
    if path.name == "scores_esm1v.tsv" and sequence and len(df) != expected_esm1v_rows(sequence):
        return f"{len(df)} rows, expected {expected_esm1v_rows(sequence)}"
    if sequence is not None and len(df):
        pos = df["position"].to_numpy()
        if pos.max() > len(sequence) or pos.min() < 1:
            return "positions outside the sequence"
    return None


def check_clinvar_table(path: Path) -> Optional[str]:
    from .clinvar import CLINVAR_COLUMNS
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as e:  # noqa: BLE001
        return f"unreadable ({e.__class__.__name__})"
    missing = set(CLINVAR_COLUMNS) - set(df.columns)
    return f"missing columns {sorted(missing)}" if missing else None


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
        elif name == "clinvar.tsv":
            prob = check_clinvar_table(p)
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
        all_problems.extend(probs)
    return all_problems


def compact_cache(cache_dir: Path, data_dir: Optional[Path] = None, accessions: Optional[list[str]] = None,
                  dry_run: bool = False, remove_zenodo_extracted: bool = False) -> dict:
    """Delete cache files that are redundant with the per-protein score tables.

    * esm1v_<k>.npy once a valid scores_esm1v.tsv exists (5 files per protein)
    * ebi_variation.json, converted to the much smaller clinvar.tsv
    * status.json (its content is in proteins.tsv)
    * thermompnn_run/ and foldx/ work directories once their score tables exist
    * ThermoMPNN tables of proteins split into several AlphaFold fragments, which older versions
      combined with wrong positions; they are rebuilt from the archive on the next run
    * leftover temporary files
    """
    import shutil

    from . import clinvar
    from .predictors.thermompnn import ZenodoThermoMPNN

    cache_dir = Path(cache_dir)
    counts = {"files_removed": 0, "clinvar_converted": 0, "thermompnn_reset": 0}

    def rm(path: Path) -> None:
        if path.is_dir():
            n = sum(1 for q in path.rglob("*") if q.is_file())
            if not dry_run:
                shutil.rmtree(path, ignore_errors=True)
        else:
            n = 1
            if not dry_run:
                path.unlink(missing_ok=True)
        counts["files_removed"] += n

    multi_fragment: set[str] = set()
    if data_dir is not None:
        zen = ZenodoThermoMPNN(Path(data_dir))
        if zen.index_path.exists():
            multi_fragment = {acc for acc in zen.build_index() if len(zen.members_for(acc)) > 1}
        if remove_zenodo_extracted and zen.extract_dir.exists():
            log.info("removing unpacked archive %s", zen.extract_dir)
            rm(zen.extract_dir)

    dirs = [cache_dir / a for a in accessions] if accessions else sorted(
        d for d in cache_dir.iterdir() if d.is_dir() and not d.name.startswith(("_", ".")) and d.name != "huggingface")
    for d in dirs:
        if not d.is_dir():
            continue
        seq = None
        try:
            seq = json.loads((d / "uniprot.json").read_text())["sequence"]["value"]
        except Exception:  # noqa: BLE001
            pass
        for p in d.iterdir():
            if p.is_file() and (p.name.endswith((".tmp", ".part")) or ".tmp." in p.name):
                rm(p)
        esm = d / "scores_esm1v.tsv"
        if esm.exists() and check_score_table(esm, seq) is None:
            for p in d.glob("esm1v_*.npy"):
                rm(p)
        legacy = d / clinvar.LEGACY_JSON
        if legacy.exists():
            if not dry_run:
                try:
                    clinvar.load_clinvar(d.name, cache_dir)   # converts and removes the JSON
                    counts["clinvar_converted"] += 1
                    counts["files_removed"] += 1
                except Exception as e:  # noqa: BLE001
                    log.warning("%s: could not convert %s (%s)", d.name, legacy, e)
            else:
                counts["clinvar_converted"] += 1
        if (d / "status.json").exists():
            rm(d / "status.json")
        for table, work in (("scores_thermompnn.tsv", "thermompnn_run"), ("scores_foldx.tsv", "foldx")):
            if (d / work).exists() and (d / table).exists() and check_score_table(d / table, seq) is None:
                rm(d / work)
        tm = d / "scores_thermompnn.tsv"
        if d.name in multi_fragment and tm.exists():
            try:
                src = pd.read_csv(tm, sep="\t", usecols=["thermompnn_source"])["thermompnn_source"].iloc[0]
            except Exception:  # noqa: BLE001
                src = "zenodo"
            if str(src).startswith("zenodo"):
                rm(tm)
                counts["thermompnn_reset"] += 1
    return counts
