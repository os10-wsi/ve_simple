"""TED (The Encyclopedia of Domains) domain assignments from ted.cathdb.info, plus verification
of each domain against the sequence of the cached AlphaFold model.

TED was computed on AFDB v4 models. Its `md5_domain` is the MD5 of the domain's sequence
(segments concatenated), so every domain can be checked against the cached (newer) model: if the
sequence changed, boundaries are shifted by up to +-MAX_SHIFT residues to find the matching
position, otherwise the domain is dropped and logged.
"""
from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)
API = "https://ted.cathdb.info/api/v1/uniprot/summary/{acc}"
TED_FIELDS = ["ted_id", "chopping", "consensus_level", "nres_domain", "num_segments", "plddt", "num_helix",
              "num_strand", "num_helix_strand_turn", "cath_label", "cath_assignment_level",
              "cath_assignment_method", "packing_density", "norm_rg", "md5_domain"]
MAX_SHIFT = 30


def fetch_one(acc: str, session: requests.Session, page: int = 100, retries: int = 4) -> Optional[list[dict]]:
    """All TED domains for `acc` (paginated); [] if TED has none; None on persistent failure."""
    out, skip = [], 0
    while True:
        for attempt in range(retries):
            try:
                r = session.get(API.format(acc=acc), params={"skip": skip, "limit": page}, timeout=60)
                if r.status_code == 404:
                    return out
                r.raise_for_status()
                d = r.json()
                break
            except Exception as ex:  # noqa: BLE001
                if attempt == retries - 1:
                    log.warning("%s: TED query failed (%s)", acc, ex)
                    return None
                time.sleep(2 ** attempt)
        data = d.get("data", [])
        for x in data:
            out.append({"accession": acc, **{k: x.get(k) for k in TED_FIELDS}})
        skip += len(data)
        if not data or skip >= int(d.get("count", 0)):
            return out


def fetch_ted(accessions: list[str], out_dir: Path, threads: int = 8, part_size: int = 500) -> None:
    """Fetch TED for accessions not yet queried; writes data parts and a 'queried' log (resumable)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set(load_queried(out_dir))
    todo = [a for a in accessions if a not in done]
    log.info("TED: %d accessions already queried, %d to fetch", len(done), len(todo))
    s = requests.Session()
    s.headers["User-Agent"] = "protshape/0.1 (python-requests)"
    n_parts = len(list(out_dir.glob("domains-*.parquet")))
    for i in range(0, len(todo), part_size):
        chunk = todo[i:i + part_size]
        with ThreadPoolExecutor(threads) as ex:
            res = list(ex.map(lambda a: (a, fetch_one(a, s)), chunk))
        rows = [r for _, doms in res if doms for r in doms]
        queried = pd.DataFrame([{"accession": a, "n_domains": len(d)} for a, d in res if d is not None])
        tag = f"{n_parts:05d}"
        if rows:
            pd.DataFrame(rows).to_parquet(out_dir / f"domains-{tag}.parquet", index=False)
        queried.to_parquet(out_dir / f"queried-{tag}.parquet", index=False)
        n_parts += 1
        log.info("TED: %d/%d fetched", min(i + part_size, len(todo)), len(todo))


def load_queried(out_dir: Path) -> list[str]:
    parts = sorted(Path(out_dir).glob("queried-*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts])["accession"].tolist() if parts else []


def load_domains(out_dir: Path) -> pd.DataFrame:
    parts = sorted(Path(out_dir).glob("domains-*.parquet"))
    if not parts:
        return pd.DataFrame(columns=["accession"] + TED_FIELDS)
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).drop_duplicates("ted_id")


def parse_chopping(chopping: str) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in seg.split("-")) for seg in str(chopping).split("_") if "-" in seg]


def verify_domain(segments: list[tuple[int, int]], md5: str, sequence: str) -> tuple[Optional[list[tuple[int, int]]], int]:
    """Return (segments in the model's numbering, shift applied) or (None, 0) if no match within MAX_SHIFT."""
    def h(segs):
        return hashlib.md5("".join(sequence[a - 1:b] for a, b in segs).encode()).hexdigest()
    if not md5:
        return segments, 0
    for shift in [0] + [s for k in range(1, MAX_SHIFT + 1) for s in (k, -k)]:
        segs = [(a + shift, b + shift) for a, b in segments]
        if segs[0][0] < 1 or segs[-1][1] > len(sequence):
            continue
        if h(segs) == md5:
            return segs, shift
    return None, 0
