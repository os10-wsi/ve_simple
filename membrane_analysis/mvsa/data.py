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
    # The shape results (and so proteins/shape_units/residues) cover every dataset that was run; keep only the
    # proteins of THIS dataset's ve_simple assembly (in_ve_results), so structural sections are not mixed across
    # datasets. Older data directories without that flag fall back to the dataset's main taxid.
    prot = out["proteins"]
    if "in_ve_results" in prot and prot["in_ve_results"].any():
        out["proteins"] = prot[prot["in_ve_results"].astype(bool)].reset_index(drop=True)
    elif out["meta"].get("taxid") is not None and "taxid" in prot:
        out["proteins"] = prot[prot["taxid"] == out["meta"]["taxid"]].reset_index(drop=True)
    keep = set(out["proteins"]["accession"])
    for t in TABLES[1:]:
        if t in out and "accession" in out[t]:
            out[t] = out[t][out[t]["accession"].isin(keep)].reset_index(drop=True)
    cls = out["proteins"].set_index("accession")["class_annotation"]
    for t in ("labelled_variants", "variant_sample", "protein_scores", "protein_auc", "shape_units", "residues"):
        if t in out and "accession" in out[t]:
            out[t]["protein_class"] = out[t]["accession"].map(cls)
    out["meta"].setdefault("dataset", d.name.removeprefix("data_") if d.name.startswith("data_") else d.name)
    return out


def dataset_label(meta: dict) -> str:
    """Readable name of a dataset: 'human', 'yeast', 'mouse', 'families', ..."""
    names = {9606: "human", 559292: "yeast", 10090: "mouse", 10116: "rat", 7955: "zebrafish", 7227: "fly",
             6239: "worm", 3702: "Arabidopsis", 284812: "fission yeast", 83333: "E. coli", 44689: "Dictyostelium"}
    ds = meta.get("dataset")
    if ds and not ds.isdigit():
        return ds
    return names.get(meta.get("taxid"), str(meta.get("taxid")))


def available_predictors(df: pd.DataFrame, min_frac: float = 0.05) -> list[str]:
    """Predictors with scores for at least `min_frac` of rows (e.g. FoldX is absent until it has run)."""
    return [p for p in PREDICTORS if p in df and df[p].notna().mean() >= min_frac]
