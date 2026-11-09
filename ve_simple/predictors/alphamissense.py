"""AlphaMissense scores from the AlphaFold DB per-protein substitution table."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_VAR_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def read_alphamissense(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    parsed = df["protein_variant"].str.extract(_VAR_RE)
    out = pd.DataFrame({
        "position": parsed[1].astype(int),
        "wt": parsed[0],
        "mut": parsed[2],
        "alphamissense": df["am_pathogenicity"].astype(float),
        "alphamissense_class": df["am_class"],
    })
    return out.dropna(subset=["position"])


# --------------------------------------------------------------------------- bulk fallback
# AlphaFold DB serves per-protein AlphaMissense tables, but not for every human protein (missing
# or 404 for some proteins whose UniProt sequence changed after 2021).  The complete release is a
# single 1.2 GB file; proteins missing from AlphaFold DB are filled from it in ONE pass and written
# to the cache in the same per-protein format.  Its 2021 residue numbering is aligned to the current
# sequence when the table is used (see seqmap).
BULK_URL = "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz"
BULK_NAME = "AlphaMissense_aa_substitutions.tsv.gz"
_BULK_CLASS = {"likely_benign": "LBen", "benign": "LBen", "ambiguous": "Amb",
               "likely_pathogenic": "LPath", "pathogenic": "LPath"}


def per_protein_path(cache_dir: Path, acc: str) -> Path:
    return Path(cache_dir) / acc / f"AF-{acc}-F1-aa-substitutions.csv"


def fill_from_bulk(accessions, cache_dir: Path, data_dir: Path, download: bool = True) -> int:
    """Write per-protein AlphaMissense tables for `accessions` from the bulk release (one pass).

    Returns the number of proteins written.  Accessions absent from the release are left alone.
    """
    import logging
    import os

    from ..utils import download_file

    log = logging.getLogger(__name__)
    wanted = set(accessions)
    if not wanted:
        return 0
    bulk = Path(data_dir) / BULK_NAME
    if not bulk.exists():
        if not download:
            log.warning("AlphaMissense bulk file not found at %s", bulk)
            return 0
        log.info("Downloading the AlphaMissense release (1.2 GB) to fill %d protein(s)", len(wanted))
        download_file(BULK_URL, bulk, timeout=3600)
    log.info("Reading AlphaMissense release for %d protein(s) missing from AlphaFold DB", len(wanted))

    pending: dict[str, list[pd.DataFrame]] = {}
    written: set[str] = set()

    def flush(acc: str) -> None:
        df = pd.concat(pending.pop(acc), ignore_index=True)
        out = pd.DataFrame({"protein_variant": df["protein_variant"],
                            "am_pathogenicity": df["am_pathogenicity"],
                            "am_class": df["am_class"].map(_BULK_CLASS).fillna(df["am_class"])})
        dest = per_protein_path(cache_dir, acc)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if acc in written and dest.exists():          # file not sorted after all: append
            out = pd.concat([pd.read_csv(dest), out], ignore_index=True)
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
        out.to_csv(tmp, index=False)
        os.replace(tmp, dest)
        written.add(acc)

    reader = pd.read_csv(bulk, sep="\t", comment="#", chunksize=2_000_000,
                         dtype={"uniprot_id": str, "protein_variant": str, "am_pathogenicity": float,
                                "am_class": str})
    for chunk in reader:
        hit = chunk[chunk["uniprot_id"].isin(wanted)]
        for acc, grp in hit.groupby("uniprot_id", sort=False):
            pending.setdefault(acc, []).append(grp)
        # the release is sorted by accession: everything before this chunk's last accession is complete
        last = chunk["uniprot_id"].iloc[-1]
        for acc in [a for a in pending if a != last]:
            flush(acc)
    for acc in list(pending):
        flush(acc)
    missing = wanted - written
    if missing:
        log.info("%d protein(s) are not in the AlphaMissense release either", len(missing))
    return len(written)
