"""Per-protein Spearman correlations between predictors, and between predictors and residue context,
over ALL possible substitutions of each protein (one pass over the ve_simple variant table).

    python -m mvsa.protcorr --ve ../results_human --data data_human

Writes data_<name>/protein_correlations.parquet: accession, a, b, rho, n_substitutions.
Scores are oriented (higher = more damaging) so a positive rho means two predictors agree.
Residue context comes from data_<name>/residues.parquet (written by mvsa.prepare): relative SASA,
burial depth, model pLDDT and whether the residue is in a TM segment.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .groups import DIRECTION, MAIN_PREDICTORS

log = logging.getLogger("mvsa.protcorr")
CONTEXT = ["rsa", "depth_mean", "plddt_model", "in_tm"]


def correlations(df: pd.DataFrame, cols: list[str], min_n: int = 30) -> pd.DataFrame:
    """Long table of pairwise Spearman rho per protein (pairwise-complete observations)."""
    rows = []
    iu = None
    for acc, g in df.groupby("accession", sort=False):
        m = g[cols].astype(float).corr(method="spearman", min_periods=min_n)
        if iu is None:
            iu = np.triu_indices(len(cols), 1)
        v = m.to_numpy()[iu]
        rows.append(pd.DataFrame({"accession": acc, "a": np.asarray(cols)[iu[0]], "b": np.asarray(cols)[iu[1]],
                                  "rho": v, "n_substitutions": len(g)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["accession", "a", "b", "rho", "n_substitutions"])


def run(ve: Path, data: Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    path = ve / "variant_scores.tsv"
    header = pd.read_csv(path, sep="\t", nrows=0).columns
    preds = [p for p in MAIN_PREDICTORS if p in header]
    res = pd.read_parquet(data / "residues.parquet", columns=["accession", "position"] + CONTEXT)
    res["in_tm"] = res["in_tm"].astype(float)
    cols = preds + CONTEXT
    out, carry, n_rows, t0 = [], None, 0, time.time()
    for chunk in pd.read_csv(path, sep="\t", usecols=["accession", "position"] + preds, chunksize=chunksize):
        n_rows += len(chunk)
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last = chunk["accession"].iat[-1]
        carry = chunk[chunk["accession"] == last]
        done = chunk[chunk["accession"] != last]
        if len(done):
            out.append(_one(done, res, preds, cols))
        log.info("%.1f M rows read (%.0f s)", n_rows / 1e6, time.time() - t0)
    if carry is not None and len(carry):
        out.append(_one(carry, res, preds, cols))
    return pd.concat(out, ignore_index=True)


def _one(df, res, preds, cols):
    for p in preds:
        df[p] = df[p] * DIRECTION[p]
    df = df.merge(res, on=["accession", "position"], how="left")
    return correlations(df, cols)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ve", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="data_<name> directory written by mvsa.prepare")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if not (args.data / "residues.parquet").exists():
        raise SystemExit(f"{args.data}/residues.parquet not found: run mvsa.prepare first")
    out = run(args.ve, args.data, args.chunksize)
    out.to_parquet(args.data / "protein_correlations.parquet", index=False)
    log.info("done: %d proteins, %d pairs -> %s", out.accession.nunique(), len(out), args.data / "protein_correlations.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
