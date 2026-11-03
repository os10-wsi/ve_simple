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

import json
import logging
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Iterable, Optional

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
        self.extract_dir = self.data_dir / "thermompnn_outs"
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
        return acc in self.build_index()

    # -- extraction ---------------------------------------------------------------------------
    def _local_path(self, member: str) -> Path:
        return self.extract_dir / member

    def extract(self, accessions: Iterable[str]) -> dict[str, list[Path]]:
        """Extract members for `accessions` that are not yet on disk (single streaming pass)."""
        index = self.build_index()
        wanted: dict[str, str] = {}
        found: dict[str, list[Path]] = {}
        for acc in accessions:
            for member in index.get(acc, []):
                p = self._local_path(member)
                found.setdefault(acc, []).append(p)
                if not p.exists():
                    wanted[member] = acc
        if wanted:
            log.info("Extracting ThermoMPNN predictions for %d file(s) from the Zenodo archive", len(wanted))
            with tarfile.open(self.archive, "r|gz") as tf:
                for m in tf:
                    if m.name in wanted and m.isfile():
                        tf.extract(m, path=self.extract_dir, filter="data")
                        del wanted[m.name]
                        if not wanted:
                            break
        return found

    def extract_all(self) -> None:
        with tarfile.open(self.archive, "r|gz") as tf:
            tf.extractall(path=self.extract_dir, filter="data")

    # -- parsing -------------------------------------------------------------------------------
    def load(self, acc: str, sequence: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Predictions for `acc` as (position, wt, mut, thermompnn) with 1-based positions.

        The archive files were produced by ThermoMPNN's `custom_inference.py`, whose `position`
        column is a 0-based sequence index; the offset is verified against `sequence` when given.
        """
        paths = self.extract([acc]).get(acc)
        if not paths:
            return None
        frames = [parse_prediction_file(p) for p in paths]
        frames = [f for f in frames if f is not None and len(f)]
        if not frames:
            return None
        df = pd.concat(frames).drop_duplicates(["position", "wt", "mut"])
        df["position"] = df["position"] + detect_position_offset(df, sequence)
        return df[THERMOMPNN_COLUMNS].reset_index(drop=True)


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
