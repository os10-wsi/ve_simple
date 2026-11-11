"""Index the ve_simple per-protein cache (cache/<accession>/...) by UniProt accession.

Layout written by ve_simple (read-only here):
    cache/<ACC>/uniprot.json                 full UniProtKB JSON entry
    cache/<ACC>/afdb.json                    AlphaFold DB API record(s) for the accession
    cache/<ACC>/AF-<ACC>-F1-model.pdb        AlphaFold model (F1), latest AFDB version at download time
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)
SKIP_DIRS = {"huggingface"}


def _index_one(d: Path) -> Optional[dict]:
    up = d / "uniprot.json"
    if not up.exists():
        return None
    try:
        e = json.loads(up.read_text())
    except Exception as ex:  # noqa: BLE001
        log.warning("%s: unreadable uniprot.json (%s)", d.name, ex)
        return {"accession": d.name, "index_error": f"uniprot.json: {ex}"}
    acc = e.get("primaryAccession", d.name)
    row = {
        "accession": acc, "cache_dir": str(d), "taxid": int(e.get("organism", {}).get("taxonId", 0)),
        "organism": e.get("organism", {}).get("scientificName", ""),
        "reviewed": e.get("entryType", "").startswith("UniProtKB reviewed"),
        "uniprot_length": int(e.get("sequence", {}).get("length", 0)), "index_error": "",
    }
    pdbs = sorted(d.glob(f"AF-{acc}-F*-model*.pdb"))
    row["model_path"] = str(pdbs[0]) if pdbs else ""
    row["has_model"] = bool(pdbs)
    row["model_entity"], row["model_version"], row["model_length"] = "", -1, 0
    afdb = d / "afdb.json"
    if afdb.exists():
        try:
            entries = json.loads(afdb.read_text()) or []
            ent = next((x for x in entries if x.get("modelEntityId") == f"AF-{acc}-F1"), entries[0] if entries else None)
            if ent:
                row["model_entity"] = ent.get("modelEntityId", "")
                row["model_version"] = int(ent.get("latestVersion", -1))
                row["model_length"] = int(ent.get("sequenceEnd", 0)) - int(ent.get("sequenceStart", 1)) + 1
        except Exception as ex:  # noqa: BLE001
            row["index_error"] = f"afdb.json: {ex}"
    # AFDB splits >2700-residue human proteins into fragments; only F1 is used, and flagged
    row["is_fragment"] = bool(row["has_model"] and row["model_length"] and row["model_length"] < row["uniprot_length"])
    return row


def index_cache(cache_dir: Path, accessions: Optional[list[str]] = None, threads: int = 16) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    if accessions is None:
        dirs = [d for d in cache_dir.iterdir() if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith(".")]
    else:
        dirs = [cache_dir / a for a in accessions]
    with ThreadPoolExecutor(threads) as ex:
        rows = [r for r in ex.map(_index_one, dirs) if r is not None]
    return pd.DataFrame(rows)


def read_id_file(path: Path) -> list[str]:
    """First UniProt accession on each line (plain list or UniProt TSV download)."""
    import re

    acc_re = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$")
    out, seen = [], set()
    for line in Path(path).read_text().splitlines():
        for tok in re.split(r"[\s,;|>]+", line.split("#")[0].strip()):
            tok = tok.split("-")[0]
            if acc_re.match(tok):
                if tok not in seen:
                    seen.add(tok)
                    out.append(tok)
                break
    return out
