"""Shared helpers: HTTP session with retries, caching, logging."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import __version__

log = logging.getLogger("ve_simple")

USER_AGENT = f"ve_simple/{__version__} (python-requests; https://github.com/os10-wsi/ve_simple)"


def get_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["GET", "HEAD"]))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = USER_AGENT
    return s


_SESSION: Optional[requests.Session] = None


def session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = get_session()
    return _SESSION


def http_get(url: str, timeout: int = 120, **kw) -> requests.Response:
    r = session().get(url, timeout=timeout, **kw)
    return r


def download_file(url: str, dest: Path, timeout: int = 600, resume: bool = True) -> Path:
    """Stream a (possibly large) file to disk, resuming partial downloads if possible."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {}
    mode = "wb"
    if resume and tmp.exists() and tmp.stat().st_size > 0:
        headers["Range"] = f"bytes={tmp.stat().st_size}-"
        mode = "ab"
    with session().get(url, stream=True, timeout=timeout, headers=headers) as r:
        if r.status_code == 416:  # already complete
            tmp.rename(dest)
            return dest
        if r.status_code == 200 and mode == "ab":
            mode = "wb"  # server ignored the range request
        r.raise_for_status()
        with open(tmp, mode) as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    tmp.rename(dest)
    return dest


def read_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def write_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def normalise_accession(acc: str) -> str:
    """Strip whitespace and isoform suffixes (P04637-2 -> P04637)."""
    acc = acc.strip()
    if "-" in acc:
        base = acc.split("-")[0]
        log.warning("Isoform accession %s given; using canonical %s", acc, base)
        acc = base
    return acc.upper()


def read_id_list(path: Path) -> list[str]:
    ids: list[str] = []
    seen = set()
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            for tok in line.replace(",", " ").split():
                acc = normalise_accession(tok)
                if acc not in seen:
                    seen.add(acc)
                    ids.append(acc)
    return ids


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        log.info("%s took %.1fs", self.label, time.time() - self.t0)
