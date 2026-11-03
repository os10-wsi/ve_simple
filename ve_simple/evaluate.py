"""ROC-AUC of every predictor for pathogenic-vs-benign discrimination, per protein.

AUC standard errors use the DeLong (1988) method; confidence intervals are from a stratified
bootstrap (default 1000 resamples) when enough variants are available, otherwise DeLong normal CI.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from .constants import PREDICTOR_DIRECTIONS

log = logging.getLogger(__name__)

AUC_COLUMNS = ["accession", "predictor", "n_pathogenic", "n_benign", "n_scored", "auc", "auc_se",
               "ci_low", "ci_high", "ci_method", "p_value", "direction"]


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_auc_variance(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """AUC and its DeLong variance for scores of positives (`pos`) and negatives (`neg`)."""
    m, n = len(pos), len(neg)
    scores = np.concatenate([pos, neg])
    tx = _compute_midrank(pos)
    ty = _compute_midrank(neg)
    tz = _compute_midrank(scores)
    auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
    v10 = tz[:m] - tx           # structural components for positives
    v01 = 1.0 - (tz[m:] - ty)   # structural components for negatives
    s10 = np.var(v10, ddof=1) if m > 1 else 0.0
    s01 = np.var(v01, ddof=1) if n > 1 else 0.0
    var = s10 / m + s01 / n
    return float(auc), float(var)


def bootstrap_ci(pos: np.ndarray, neg: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        p = rng.choice(pos, size=len(pos), replace=True)
        q = rng.choice(neg, size=len(neg), replace=True)
        y = np.r_[np.ones(len(p)), np.zeros(len(q))]
        aucs[b] = roc_auc_score(y, np.r_[p, q])
    return float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2))


def auc_for_protein(variants: pd.DataFrame, accession: str, predictors: Optional[list[str]] = None,
                    n_boot: int = 1000, min_per_class: int = 1, seed: int = 0) -> pd.DataFrame:
    """One row per predictor with AUC, SE, CI for `variants` (needs `clinvar_label` column)."""
    predictors = predictors or [p for p in PREDICTOR_DIRECTIONS if p in variants.columns]
    lab = variants["clinvar_label"]
    labelled = variants[lab.isin(["pathogenic", "benign"])]
    rows = []
    for pred in predictors:
        direction = PREDICTOR_DIRECTIONS.get(pred, 1)
        sub = labelled[["clinvar_label", pred]].dropna()
        y = (sub["clinvar_label"] == "pathogenic").to_numpy()
        s = sub[pred].to_numpy(dtype=float) * direction
        n_pos, n_neg = int(y.sum()), int((~y).sum())
        row = {"accession": accession, "predictor": pred, "n_pathogenic": n_pos, "n_benign": n_neg,
               "n_scored": len(sub), "auc": np.nan, "auc_se": np.nan, "ci_low": np.nan,
               "ci_high": np.nan, "ci_method": "", "p_value": np.nan, "direction": direction}
        if n_pos >= min_per_class and n_neg >= min_per_class:
            auc, var = delong_auc_variance(s[y], s[~y])
            se = float(np.sqrt(var))
            row.update(auc=auc, auc_se=se)
            if n_boot and n_pos >= 2 and n_neg >= 2:
                lo, hi = bootstrap_ci(s[y], s[~y], n_boot=n_boot, seed=seed)
                row.update(ci_low=lo, ci_high=hi, ci_method=f"bootstrap{n_boot}")
            elif se > 0:
                z = stats.norm.ppf(0.975)
                row.update(ci_low=max(0.0, auc - z * se), ci_high=min(1.0, auc + z * se), ci_method="delong")
            if se > 0:
                zstat = (auc - 0.5) / se
                row["p_value"] = float(2 * stats.norm.sf(abs(zstat)))
        rows.append(row)
    return pd.DataFrame(rows, columns=AUC_COLUMNS)


def roc_points(variants: pd.DataFrame, pred: str) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    direction = PREDICTOR_DIRECTIONS.get(pred, 1)
    sub = variants[variants["clinvar_label"].isin(["pathogenic", "benign"])][["clinvar_label", pred]].dropna()
    y = (sub["clinvar_label"] == "pathogenic").to_numpy()
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    s = sub[pred].to_numpy(dtype=float) * direction
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, float(roc_auc_score(y, s))


def wide_auc_table(auc_long: pd.DataFrame, proteins: pd.DataFrame) -> pd.DataFrame:
    """One row per protein; columns <pred>_auc, <pred>_se, <pred>_ci_low, <pred>_ci_high, counts."""
    if auc_long.empty:
        return proteins.copy()
    base = auc_long.groupby("accession")[["n_pathogenic", "n_benign"]].max()
    pieces = [base]
    for stat in ("auc", "auc_se", "ci_low", "ci_high", "n_scored"):
        p = auc_long.pivot(index="accession", columns="predictor", values=stat)
        p.columns = [f"{c}_{stat}" for c in p.columns]
        pieces.append(p)
    wide = pd.concat(pieces, axis=1)
    # order columns predictor by predictor
    ordered = ["n_pathogenic", "n_benign"]
    for pred in [p for p in PREDICTOR_DIRECTIONS if f"{p}_auc" in wide.columns]:
        ordered += [f"{pred}_{s}" for s in ("auc", "auc_se", "ci_low", "ci_high", "n_scored")]
    wide = wide[ordered].reset_index()
    meta_cols = [c for c in ("accession", "gene", "entry_name", "organism", "length") if c in proteins.columns]
    return proteins[meta_cols].merge(wide, on="accession", how="left")
