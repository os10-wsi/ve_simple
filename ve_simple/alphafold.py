"""AlphaFold DB access: predicted structure (PDB) and per-protein AlphaMissense annotations."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import THREE_TO_ONE
from .utils import download_file, http_get, read_json, write_json

log = logging.getLogger(__name__)

AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"


def fetch_afdb_metadata(acc: str, cache_dir: Path) -> Optional[dict]:
    cache = Path(cache_dir) / acc / "afdb.json"
    if cache.exists():
        d = read_json(cache)
    else:
        r = http_get(AFDB_API.format(acc=acc), headers={"Accept": "application/json"})
        if r.status_code == 404:
            log.warning("No AlphaFold DB entry for %s", acc)
            write_json([], cache)
            return None
        r.raise_for_status()
        d = r.json()
        write_json(d, cache)
    if not d:
        return None
    # The API may list isoform models (AF-P04637-2-F1) and fragments; prefer the canonical F1 model.
    canonical = [e for e in d if e.get("modelEntityId") == f"AF-{acc}-F1"]
    if canonical:
        return canonical[0]
    same_acc = [e for e in d if e.get("uniprotAccession") == acc]
    pool = same_acc or d
    return sorted(pool, key=lambda e: (e.get("sequenceStart", 0), e.get("modelEntityId", "")))[0]


def fetch_structure(acc: str, cache_dir: Path, meta: Optional[dict] = None) -> Optional[Path]:
    """Download the AlphaFold PDB model for `acc`; returns the local path or None."""
    dest = Path(cache_dir) / acc / f"AF-{acc}-F1-model.pdb"
    if dest.exists():
        return dest
    meta = meta or fetch_afdb_metadata(acc, cache_dir)
    if not meta or not meta.get("pdbUrl"):
        return None
    download_file(meta["pdbUrl"], dest, resume=False)
    return dest


def fetch_alphamissense(acc: str, cache_dir: Path, meta: Optional[dict] = None) -> Optional[Path]:
    """Download the per-protein AlphaMissense substitution table (human proteins only)."""
    dest = Path(cache_dir) / acc / f"AF-{acc}-F1-aa-substitutions.csv"
    if dest.exists():
        return dest if dest.stat().st_size > 0 else None
    meta = meta or fetch_afdb_metadata(acc, cache_dir)
    url = (meta or {}).get("amAnnotationsUrl")
    if not url:
        return None
    r = http_get(url)
    if r.status_code != 200:
        log.warning("AlphaMissense table unavailable for %s (HTTP %s)", acc, r.status_code)
        dest.write_bytes(b"")
        return None
    dest.write_bytes(r.content)
    return dest


def read_pdb_chain(pdb_path: Path, chain_id: Optional[str] = None):
    """Return (Bio.PDB chain, sequence, residue numbers, pLDDT array) for the first/selected chain."""
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure("af", str(pdb_path))
    model = next(structure.get_models())
    chains = list(model.get_chains())
    chain = chains[0] if chain_id is None else model[chain_id]
    seq, resnums, plddt = [], [], []
    for res in chain:
        if res.id[0] != " ":
            continue
        one = THREE_TO_ONE.get(res.get_resname(), "X")
        seq.append(one)
        resnums.append(res.id[1])
        ca = res["CA"] if "CA" in res else next(iter(res))
        plddt.append(ca.get_bfactor())
    return chain, "".join(seq), np.asarray(resnums), np.asarray(plddt, dtype=float)
