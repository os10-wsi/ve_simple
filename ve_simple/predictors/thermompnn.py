"""ThermoMPNN ddG predictions.

Two sources, in order of preference:
1. The human proteome-wide ThermoMPNN predictions released with Liao & Lehner (2026),
   Zenodo record 18381534 (`thermompnn_outs.tar.gz`).  The archive is downloaded on demand into
   `data/zenodo/`, and the members for the requested proteins are extracted in one streaming
   pass into `data/zenodo/thermompnn_outs/`.
2. De novo inference with the ThermoMPNN code (Dieckhaus et al. 2024) on the AlphaFold model,
   for proteins absent from the archive (e.g. non-human proteins).
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from ..constants import AMINO_ACIDS
from ..utils import download_file

log = logging.getLogger(__name__)

ZENODO_RECORD = "18381534"
ZENODO_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/thermompnn_outs.tar.gz/content"
ZENODO_ARCHIVE_NAME = "thermompnn_outs.tar.gz"
THERMOMPNN_GIT = "https://github.com/Kuhlman-Lab/ThermoMPNN.git"

THERMOMPNN_COLUMNS = ["position", "wt", "mut", "thermompnn"]
_ACC_RE = re.compile(r"(?<![A-Z0-9])([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?![A-Z0-9])")


# --------------------------------------------------------------------------------------- Zenodo
class ZenodoThermoMPNN:
    """Access to the proteome-wide ThermoMPNN archive from Zenodo record 18381534."""

    def __init__(self, data_dir: Path, archive: Optional[Path] = None):
        self.data_dir = Path(data_dir)
        self.archive = Path(archive) if archive else self.data_dir / ZENODO_ARCHIVE_NAME
        self.extract_dir = self.data_dir / "thermompnn_outs"   # only used by the legacy --extract-all
        self.index_path = self.data_dir / "thermompnn_outs.index.json"
        self._index: Optional[dict[str, list[str]]] = None

    # -- download / index -------------------------------------------------------------------
    def ensure_archive(self, download: bool = True) -> Optional[Path]:
        if self.archive.exists():
            return self.archive
        if not download:
            log.warning("ThermoMPNN Zenodo archive not found at %s", self.archive)
            return None
        log.info("Downloading ThermoMPNN proteome predictions (3.3 GB) from Zenodo record %s", ZENODO_RECORD)
        download_file(ZENODO_URL, self.archive, timeout=3600)
        return self.archive

    def build_index(self, force: bool = False) -> dict[str, list[str]]:
        """Map UniProt accession -> archive member names (one streaming pass; cached as JSON)."""
        if self._index is not None and not force:
            return self._index
        if self.index_path.exists() and not force:
            self._index = json.loads(self.index_path.read_text())
            return self._index
        if not self.archive.exists():
            self._index = {}
            return self._index
        log.info("Indexing %s (single pass through the archive)", self.archive)
        index: dict[str, list[str]] = {}
        with tarfile.open(self.archive, "r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                for acc in set(_ACC_RE.findall(os.path.basename(m.name))):
                    index.setdefault(acc, []).append(m.name)
        self.index_path.write_text(json.dumps(index))
        self._index = index
        return index

    def available(self, acc: str) -> bool:
        return bool(self.members_for(acc))

    def members_for(self, acc: str) -> list[str]:
        """Canonical-model archive members for `acc` (one per AlphaFold fragment F1, F2, ...)."""
        out = []
        for name in self.build_index().get(acc, []):
            frag = member_fragment(name)
            if frag is not None and frag[0] == acc:
                out.append(name)
        return out

    def stream(self, accessions: Iterable[str], sink: Callable[[str, pd.DataFrame], None]) -> int:
        """Read the predictions for `accessions` in ONE pass over the archive, without unpacking it.

        For each protein, `sink(accession, table)` is called with (position, wt, mut, thermompnn) in
        1-based UniProt residue numbering.  Returns the number of proteins delivered.
        """
        wanted: dict[str, str] = {}
        remaining: dict[str, int] = {}
        for acc in accessions:
            members = self.members_for(acc)
            if members:
                remaining[acc] = len(members)
                for m in members:
                    wanted[m] = acc
        if not wanted:
            return 0
        log.info("Reading ThermoMPNN predictions for %d protein(s) from the Zenodo archive", len(remaining))
        frames: dict[str, list[pd.DataFrame]] = {}
        delivered = 0
        with tarfile.open(self.archive, "r|gz") as tf:
            for m in tf:
                acc = wanted.pop(m.name, None)
                if acc is None or not m.isfile():
                    continue
                fh = tf.extractfile(m)
                df = parse_prediction_bytes(fh.read(), m.name) if fh is not None else None
                if df is not None and len(df):
                    df["position"] = df["position"] + fragment_offset(member_fragment(m.name)[1])
                    frames.setdefault(acc, []).append(df)
                remaining[acc] -= 1
                if remaining[acc] == 0:
                    if frames.get(acc):
                        sink(acc, combine_fragments(frames.pop(acc)))
                        delivered += 1
                    del remaining[acc]
                if not wanted:
                    break
        for acc in list(remaining):           # members listed in the index but not found
            if frames.get(acc):
                sink(acc, combine_fragments(frames.pop(acc)))
                delivered += 1
        return delivered


# AlphaFold DB splits human proteins longer than 2,700 residues into 1,400-residue fragments
# shifted by 200 residues (F1 = 1-1400, F2 = 201-1600, ...).  ThermoMPNN's `position` column is a
# 0-based index into the fragment model, so residue = position + 1 + 200 * (fragment - 1).
FRAGMENT_STEP = 200
_MEMBER_RE = re.compile(r"AF-([A-Z0-9]+)-F(\d+)-model")


def member_fragment(name: str) -> Optional[tuple[str, int]]:
    """('P04637', 1) for '.../ThermoMPNN_inference_AF-P04637-F1-model_v4.csv'; None for isoforms."""
    m = _MEMBER_RE.search(os.path.basename(name))
    return (m.group(1), int(m.group(2))) if m else None


def fragment_offset(fragment: int) -> int:
    return 1 + FRAGMENT_STEP * (fragment - 1)


def combine_fragments(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge fragment tables; residues covered by overlapping fragments get the mean ddG."""
    df = pd.concat(frames, ignore_index=True)
    if len(frames) > 1:
        df = df.groupby(["position", "wt", "mut"], as_index=False)["thermompnn"].mean()
    return df[THERMOMPNN_COLUMNS].sort_values(["position", "mut"]).reset_index(drop=True)


def wt_agreement(df: pd.DataFrame, sequence: str, shift: int = 0) -> float:
    pos = df["position"].to_numpy() + shift
    ok = (pos >= 1) & (pos <= len(sequence))
    if ok.sum() == 0:
        return 0.0
    return float(np.mean([sequence[p - 1] == w for p, w in zip(pos[ok], df["wt"].to_numpy()[ok])]))


def detect_position_offset(df: pd.DataFrame, sequence: Optional[str], default: int = 1) -> int:
    """Return the shift that maps the file's `position` column onto 1-based residue numbers.

    Tries offsets 0 and 1 and keeps the one for which the wild-type residues agree best with
    `sequence`; without a sequence the ThermoMPNN convention (0-based, offset 1) is assumed.
    """
    if not sequence or len(df) == 0:
        return default
    best, best_frac = default, -1.0
    for off in (1, 0):
        pos = df["position"].to_numpy() + off
        ok = (pos >= 1) & (pos <= len(sequence))
        if ok.sum() == 0:
            continue
        frac = float(np.mean([sequence[p - 1] == w for p, w in zip(pos[ok], df["wt"].to_numpy()[ok])]))
        if frac > best_frac:
            best, best_frac = off, frac
    if best_frac < 0.9:
        log.warning("ThermoMPNN wild-type residues agree with the UniProt sequence at only %.0f%% of positions", 100 * best_frac)
    return best


def _standardise(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Map the various column namings seen in ThermoMPNN outputs onto (position, wt, mut, thermompnn)."""
    cols = {c.lower(): c for c in df.columns}
    ddg = next((cols[c] for c in ("ddg_pred", "ddg (kcal/mol)", "ddg", "thermompnn", "pred_ddg", "ddg_thermompnn") if c in cols), None)
    if ddg is None:
        return None
    if {"position", "wildtype", "mutation"} <= set(cols):
        out = pd.DataFrame({"position": df[cols["position"]], "wt": df[cols["wildtype"]],
                            "mut": df[cols["mutation"]], "thermompnn": df[ddg]})
    elif {"position", "wt", "mut"} <= set(cols):
        out = pd.DataFrame({"position": df[cols["position"]], "wt": df[cols["wt"]],
                            "mut": df[cols["mut"]], "thermompnn": df[ddg]})
    else:
        mcol = next((cols[c] for c in ("mutation", "mutant", "variant", "mut") if c in cols), None)
        if mcol is None:
            return None
        parsed = df[mcol].astype(str).str.extract(r"^([A-Z])(\d+)([A-Z])$")
        out = pd.DataFrame({"position": parsed[1], "wt": parsed[0], "mut": parsed[2], "thermompnn": df[ddg]})
    out = out.dropna(subset=["position"])
    out["position"] = out["position"].astype(int)
    out["thermompnn"] = pd.to_numeric(out["thermompnn"], errors="coerce")
    out = out[(out["wt"] != out["mut"]) & out["mut"].isin(list(AMINO_ACIDS))]
    return out


def parse_prediction_bytes(data: bytes, name: str = "") -> Optional[pd.DataFrame]:
    """Parse one ThermoMPNN prediction file held in memory (as read from the tar archive)."""
    try:
        df = pd.read_csv(io.BytesIO(data), sep="\t" if ".tsv" in name else ",")
    except Exception as e:  # noqa: BLE001
        log.warning("Could not parse %s: %s", name, e)
        return None
    return _standardise(df)


def parse_prediction_file(path: Path) -> Optional[pd.DataFrame]:
    """Parse one ThermoMPNN prediction file (CSV/TSV, optionally gzipped) in any known layout."""
    path = Path(path)
    sep = "\t" if ".tsv" in path.name or ".txt" in path.name else ","
    try:
        df = pd.read_csv(path, sep=sep, compression="infer")
    except Exception as e:  # noqa: BLE001
        log.warning("Could not parse %s: %s", path, e)
        return None
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=None, engine="python", compression="infer")
    return _standardise(df)


# --------------------------------------------------------------------------------------- de novo
def ensure_thermompnn_code(external_dir: Path) -> Path:
    repo = Path(external_dir) / "ThermoMPNN"
    if not (repo / "analysis" / "custom_inference.py").exists():
        log.info("Cloning ThermoMPNN into %s", repo)
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", THERMOMPNN_GIT, str(repo)], check=True)
    return repo


_RUNNER = r'''
import sys, types, runpy
# ThermoMPNN's training module imports wandb at module level; stub it for inference.
wandb = types.ModuleType("wandb"); wandb.init = lambda *a, **k: None; wandb.log = lambda *a, **k: None
sys.modules["wandb"] = wandb
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
'''


def run_thermompnn(pdb_path: Path, out_dir: Path, external_dir: Path, chain: str = "A") -> Optional[pd.DataFrame]:
    """Run ThermoMPNN site-saturation inference on a PDB; returns (position, wt, mut, thermompnn)."""
    repo = ensure_thermompnn_code(external_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = Path(pdb_path).resolve()
    pdb_id = pdb_path.stem
    csv = out_dir / f"ThermoMPNN_inference_{pdb_id}.csv"
    if not csv.exists():
        script = repo / "analysis" / "custom_inference.py"
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{repo}{os.pathsep}{repo / 'analysis'}{os.pathsep}" + env.get("PYTHONPATH", "")
        cmd = [sys.executable, "-c", _RUNNER, str(script), "--pdb", str(pdb_path), "--chain", chain,
               "--out_dir", str(out_dir)]
        log.info("Running ThermoMPNN on %s", pdb_path)
        res = subprocess.run(cmd, cwd=str(repo), env=env, capture_output=True, text=True)
        if res.returncode != 0 or not csv.exists():
            log.error("ThermoMPNN failed for %s:\n%s", pdb_path, res.stderr[-4000:])
            return None
    df = pd.read_csv(csv)
    # custom_inference.py reports 0-based sequence indices; convert to 1-based residue numbers.
    out = pd.DataFrame({"position": df["position"].astype(int) + 1, "wt": df["wildtype"],
                        "mut": df["mutation"], "thermompnn": pd.to_numeric(df["ddG_pred"], errors="coerce")})
    out = out[out["wt"] != out["mut"]].reset_index(drop=True)
    return out[THERMOMPNN_COLUMNS]
