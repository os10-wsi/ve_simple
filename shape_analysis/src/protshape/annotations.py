"""UniProt annotations relevant to membrane classification, read from the cached uniprot.json."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _seg(f):
    """(start, end) of a feature, or None if either end is unknown in UniProt."""
    a, b = f["location"]["start"].get("value"), f["location"]["end"].get("value")
    return (int(a), int(b)) if a is not None and b is not None else None


def parse_entry(e: dict) -> dict:
    feats = e.get("features", [])
    # Every TM/intramembrane feature counts towards the class; only those with both ends known give
    # residue positions (e.g. TIM23 P32897 has a TM segment with an unknown end).
    tm = [f for f in feats if f.get("type") == "Transmembrane"]
    tm_kind = ["beta" if "beta" in (f.get("description") or "").lower() else "helical" for f in tm]
    tm_segs = [_seg(f) for f in tm if _seg(f)]
    intra = [f for f in feats if f.get("type") == "Intramembrane"]
    intra_segs = [_seg(f) for f in intra if _seg(f)]
    lipid = [f for f in feats if f.get("type") == "Lipidation"]
    locs, topos = [], []
    for c in e.get("comments", []):
        if c.get("commentType") == "SUBCELLULAR LOCATION":
            for sl in c.get("subcellularLocations", []):
                locs.append(sl.get("location", {}).get("value", ""))
                if sl.get("topology"):
                    topos.append(sl["topology"].get("value", ""))
    similarity = " | ".join(t.get("value", "") for c in e.get("comments", []) if c.get("commentType") == "SIMILARITY"
                            for t in c.get("texts", []))
    go_cc = [p["value"][2:] for x in e.get("uniProtKBCrossReferences", []) if x.get("database") == "GO"
             for p in x.get("properties", []) if p.get("key") == "GoTerm" and p.get("value", "").startswith("C:")]
    desc = e.get("proteinDescription", {})
    rec = desc.get("recommendedName") or (desc.get("submissionNames") or [{}])[0]
    genes = e.get("genes", [])
    return {
        "accession": e["primaryAccession"],
        "reviewed": e.get("entryType", "").startswith("UniProtKB reviewed"),
        "taxid": int(e.get("organism", {}).get("taxonId", 0)),
        "gene": genes[0].get("geneName", {}).get("value", "") if genes else "",
        "protein_name": rec.get("fullName", {}).get("value", "") if rec else "",
        "length": int(e.get("sequence", {}).get("length", 0)),
        "mass": float(e.get("sequence", {}).get("molWeight", 0)),
        "tm_start": [a for a, _ in tm_segs], "tm_end": [b for _, b in tm_segs], "tm_kind": tm_kind,
        "n_tm_unplaced": len(tm) - len(tm_segs),
        "n_tm_helical": tm_kind.count("helical"), "n_tm_beta": tm_kind.count("beta"),
        "intramem_start": [a for a, _ in intra_segs], "intramem_end": [b for _, b in intra_segs],
        "n_intramem": len(intra), "n_lipid": len(lipid),
        "lipid_desc": "; ".join(sorted({f.get("description", "") for f in lipid})),
        "subcellular_location": "; ".join(locs), "topology": "; ".join(topos),
        "location_mentions_membrane": any("membrane" in s.lower() for s in locs + topos),
        "keywords": "; ".join(k.get("name", "") for k in e.get("keywords", [])),
        "go_cc": "; ".join(go_cc), "similarity": similarity,
        "uniprot_last_update": e.get("entryAudit", {}).get("lastAnnotationUpdateDate", ""),
        "sequence_version": e.get("entryAudit", {}).get("sequenceVersion", None),
    }


def annotate(index: pd.DataFrame, threads: int = 16) -> pd.DataFrame:
    def one(path):
        try:
            return parse_entry(json.loads((Path(path) / "uniprot.json").read_text()))
        except Exception as ex:  # noqa: BLE001
            log.warning("%s: could not parse UniProt entry (%s)", path, ex)
            return None
    with ThreadPoolExecutor(threads) as ex:
        rows = [r for r in ex.map(one, index["cache_dir"]) if r is not None]
    return pd.DataFrame(rows)
