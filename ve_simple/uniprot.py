"""UniProt REST access: sequence, organism, gene name."""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .constants import HUMAN_TAXID
from .utils import http_get, read_json, write_json

log = logging.getLogger(__name__)

UNIPROT_JSON = "https://rest.uniprot.org/uniprotkb/{acc}.json"


@dataclass
class ProteinEntry:
    accession: str
    entry_name: str
    protein_name: str
    gene: str
    organism: str
    taxid: int
    sequence: str
    reviewed: bool
    requested_accession: str = ""

    @property
    def is_human(self) -> bool:
        return self.taxid == HUMAN_TAXID

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_human"] = self.is_human
        d["length"] = len(self.sequence)
        return d


def _parse_entry(d: dict, requested: str) -> ProteinEntry:
    acc = d["primaryAccession"]
    desc = d.get("proteinDescription", {})
    name = ""
    rec = desc.get("recommendedName") or desc.get("submissionNames", [{}])[0]
    if rec:
        name = rec.get("fullName", {}).get("value", "")
    genes = d.get("genes", [])
    gene = genes[0].get("geneName", {}).get("value", "") if genes else ""
    org = d.get("organism", {})
    return ProteinEntry(
        accession=acc,
        entry_name=d.get("uniProtkbId", ""),
        protein_name=name,
        gene=gene,
        organism=org.get("scientificName", ""),
        taxid=int(org.get("taxonId", 0)),
        sequence=d["sequence"]["value"],
        reviewed=d.get("entryType", "").startswith("UniProtKB reviewed"),
        requested_accession=requested,
    )


def fetch_entry(acc: str, cache_dir: Path) -> Optional[ProteinEntry]:
    """Fetch (and cache) a UniProtKB entry. Returns None if the accession is unknown."""
    cache = Path(cache_dir) / acc / "uniprot.json"
    if cache.exists():
        d = read_json(cache)
    else:
        r = http_get(UNIPROT_JSON.format(acc=acc), headers={"Accept": "application/json"})
        if r.status_code == 404:
            log.error("UniProt accession %s not found", acc)
            return None
        r.raise_for_status()
        d = r.json()
        write_json(d, cache)
    if "primaryAccession" not in d:
        log.error("Unexpected UniProt response for %s", acc)
        return None
    entry = _parse_entry(d, acc)
    if entry.accession != acc:
        log.warning("Accession %s resolved to primary accession %s", acc, entry.accession)
    return entry
