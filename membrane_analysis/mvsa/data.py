"""Loading the prepared tables."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .groups import PREDICTORS

TABLES = ["proteins", "labelled_variants", "variant_sample", "protein_scores", "protein_auc", "shape_units", "residues"]


def load(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(data_dir)
    out = {t: pd.read_parquet(d / f"{t}.parquet") for t in TABLES if (d / f"{t}.parquet").exists()}
    out["meta"] = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    cls = out["proteins"].set_index("accession")["class_annotation"]
    for t in ("labelled_variants", "variant_sample", "protein_scores", "protein_auc", "shape_units", "residues"):
        if t in out and "accession" in out[t]:
            out[t]["protein_class"] = out[t]["accession"].map(cls)
    return out


def available_predictors(df: pd.DataFrame, min_frac: float = 0.05) -> list[str]:
    """Predictors with scores for at least `min_frac` of rows (e.g. FoldX is absent until it has run)."""
    return [p for p in PREDICTORS if p in df and df[p].notna().mean() >= min_frac]
