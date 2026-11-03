"""FoldX ddG predictions (BuildModel) on the AlphaFold model.

FoldX is licensed software and must be obtained from https://foldxsuite.crg.eu.  Point the
pipeline at the binary with `--foldx-binary` or the FOLDX_BINARY environment variable.  When the
binary is not available the `foldx` column is left empty (NaN) and a warning is logged.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

from ..alphafold import read_pdb_chain
from ..constants import AMINO_ACIDS

log = logging.getLogger(__name__)

FOLDX_COLUMNS = ["position", "wt", "mut", "foldx"]


def find_foldx(explicit: Optional[str] = None) -> Optional[str]:
    cand = explicit or os.environ.get("FOLDX_BINARY") or shutil.which("foldx") or shutil.which("FoldX")
    if cand and os.path.exists(cand):
        return str(cand)
    return None


def _run(binary: str, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    rotabase = Path(binary).parent / "rotabase.txt"  # FoldX 4 needs it next to the PDB
    if rotabase.exists() and not (cwd / "rotabase.txt").exists():
        shutil.copy(rotabase, cwd / "rotabase.txt")
    return subprocess.run([binary] + args, cwd=str(cwd), capture_output=True, text=True)


def repair_pdb(binary: str, pdb_path: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = Path(pdb_path)
    local = work_dir / pdb_path.name
    if not local.exists():
        shutil.copy(pdb_path, local)
    repaired = work_dir / f"{pdb_path.stem}_Repair.pdb"
    if repaired.exists():
        return repaired
    res = _run(binary, ["--command=RepairPDB", f"--pdb={local.name}", "--pdb-dir=.", "--output-dir=."], work_dir)
    if res.returncode != 0 or not repaired.exists():
        raise RuntimeError(f"FoldX RepairPDB failed: {res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    return repaired


def foldx_saturation(binary: str, pdb_path: Path, work_dir: Path, chain: str = "A",
                     chunk_size: int = 200, positions: Optional[list[int]] = None,
                     number_of_runs: int = 1) -> pd.DataFrame:
    """BuildModel ddG for every single substitution (or only `positions`) of the first chain."""
    work_dir = Path(work_dir)
    repaired = repair_pdb(binary, pdb_path, work_dir)
    _, seq, resnums, _ = read_pdb_chain(repaired)
    muts = []
    for aa, num in zip(seq, resnums):
        if aa not in AMINO_ACIDS or (positions is not None and int(num) not in positions):
            continue
        for mut in AMINO_ACIDS:
            if mut != aa:
                muts.append((int(num), aa, mut))
    results: dict[tuple, float] = {}
    for ci in range(0, len(muts), chunk_size):
        chunk = muts[ci:ci + chunk_size]
        tag = f"c{ci // chunk_size:05d}"
        dif = work_dir / f"Dif_{repaired.stem}_{tag}.fxout"
        if not dif.exists():
            mlist = work_dir / f"individual_list_{tag}.txt"
            mlist.write_text("".join(f"{wt}{chain}{pos}{mut};\n" for pos, wt, mut in chunk))
            res = _run(binary, ["--command=BuildModel", f"--pdb={repaired.name}", "--pdb-dir=.",
                                f"--mutant-file={mlist.name}", f"--numberOfRuns={number_of_runs}",
                                f"--output-file={repaired.stem}_{tag}", "--output-dir=."], work_dir)
            raw = work_dir / f"Dif_{repaired.stem}_{tag}.fxout"
            if res.returncode != 0 or not raw.exists():
                log.error("FoldX BuildModel failed for chunk %s: %s", tag, res.stdout[-1500:])
                continue
            # remove bulky mutant/WT model PDBs that BuildModel writes
            for p in work_dir.glob(f"*{repaired.stem}_{tag}*.pdb"):
                p.unlink()
        for (pos, wt, mut), ddg in zip(chunk, _parse_dif(dif, number_of_runs)):
            results[(pos, wt, mut)] = ddg
    rows = [{"position": p, "wt": w, "mut": m, "foldx": results.get((p, w, m))} for p, w, m in muts]
    return pd.DataFrame(rows, columns=FOLDX_COLUMNS)


def _parse_dif(path: Path, number_of_runs: int = 1) -> list[float]:
    """Total-energy differences (ddG) from a Dif_*.fxout file, averaged over runs per mutant."""
    vals: list[float] = []
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0].endswith(".pdb"):
            continue
        try:
            vals.append(float(parts[1]))
        except ValueError:
            continue
    if number_of_runs > 1:
        vals = [sum(vals[i:i + number_of_runs]) / number_of_runs for i in range(0, len(vals), number_of_runs)]
    return vals
