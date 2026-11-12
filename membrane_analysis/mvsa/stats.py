"""Statistics used throughout: AUC with protein-level (cluster) bootstrap, stratified/matched AUC,
thresholds tuned on soluble proteins, calibration, effect sizes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from .groups import DIRECTION


def labels(df: pd.DataFrame, include_likely: bool = True, min_stars: int = 0) -> pd.DataFrame:
    """Pathogenic (1) / benign (0) variants from the fine ClinVar class, with an optional review filter.

    min_stars: 0 = all; 1 = drop 'no assertion criteria provided' / 'no classification provided';
    2 = require 'multiple submitters' or 'expert panel' / 'practice guideline'.
    """
    pos = {"pathogenic"} | ({"likely_pathogenic"} if include_likely else set())
    neg = {"benign"} | ({"likely_benign"} if include_likely else set())
    d = df[df["clinvar_class"].isin(pos | neg)].copy()
    if min_stars:
        rs = d["clinvar_review_status"].fillna("").str.lower()
        ok = ~rs.str.contains("no assertion|no classification") & (rs != "")
        if min_stars >= 2:
            ok &= rs.str.contains("multiple submitters|expert panel|practice guideline")
        d = d[ok]
    d["y"] = d["clinvar_class"].isin(pos).astype(int)
    return d


def oriented(df, p):
    return df[p].astype(float) * DIRECTION[p]


def auc(y, s, w=None) -> float:
    y, s = np.asarray(y), np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    w = None if w is None else np.asarray(w)[ok]
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s, sample_weight=w))


def cluster_bootstrap(df: pd.DataFrame, stat, n_boot: int = 300, seed: int = 0, group: str = "accession") -> np.ndarray:
    """Re-sample whole proteins with replacement and recompute `stat(df)`; variants of one protein are not
    independent, so this is the honest uncertainty for pooled statistics."""
    rng = np.random.default_rng(seed)
    groups = df[group].to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_of = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[i] for i in pick])
        out[b] = stat(df.iloc[idx])
    return out


def pooled_auc(df: pd.DataFrame, predictor: str, n_boot: int = 300, seed: int = 0) -> dict:
    """AUC over all labelled variants of a set of proteins, with a protein-bootstrap 95% CI."""
    d = df[["accession", "y", predictor]].dropna()
    s = d[predictor].to_numpy(float) * DIRECTION[predictor]
    d = d.assign(s=s)
    point = auc(d.y, d.s)
    out = {"predictor": predictor, "n_proteins": d.accession.nunique(), "n_pathogenic": int(d.y.sum()),
           "n_benign": int((1 - d.y).sum()), "auc": point, "ci_low": np.nan, "ci_high": np.nan, "boot": None}
    if np.isfinite(point) and n_boot and d.accession.nunique() >= 5:
        b = cluster_bootstrap(d, lambda x: auc(x.y, x.s), n_boot, seed)
        b = b[np.isfinite(b)]
        out.update(ci_low=float(np.quantile(b, 0.025)), ci_high=float(np.quantile(b, 0.975)), boot=b)
    return out


def delta(a_boot, b_boot, a, b) -> dict:
    """Difference a - b with a CI and two-sided p from independent bootstrap distributions."""
    if a_boot is None or b_boot is None:
        return {"delta": a - b, "delta_low": np.nan, "delta_high": np.nan, "p": np.nan}
    n = min(len(a_boot), len(b_boot))
    d = a_boot[:n] - b_boot[:n]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"delta": a - b, "delta_low": float(np.quantile(d, 0.025)), "delta_high": float(np.quantile(d, 0.975)),
            "p": float(min(1.0, max(p, 1 / n)))}


def stratified_auc(df: pd.DataFrame, predictor: str, strata: list[str], reference: pd.DataFrame) -> float:
    """AUC computed within strata and averaged with the reference set's stratum weights (n_pos x n_neg).

    Used to ask: what would this set's AUC be if its variants had the reference set's mix of strata?
    """
    s = df[predictor].astype(float) * DIRECTION[predictor]
    d = df.assign(s=s).dropna(subset=["s"])
    ref_w = reference.groupby(strata).y.agg(lambda y: y.sum() * (len(y) - y.sum()))
    tot_w, tot = 0.0, 0.0
    for key, g in d.groupby(strata):
        if len(strata) == 1 and isinstance(key, tuple):     # single-column groupby yields 1-tuples
            key = key[0]
        w = ref_w.get(key, 0)
        a = auc(g.y, g.s)
        if w > 0 and np.isfinite(a):
            tot += w * a
            tot_w += w
    return tot / tot_w if tot_w else np.nan


def youden_threshold(y, s) -> float:
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[np.argmax(tpr - fpr)])


def error_rates(y, s, thr) -> dict:
    y, s = np.asarray(y), np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    call = s >= thr
    tp, fp = int((call & (y == 1)).sum()), int((call & (y == 0)).sum())
    fn, tn = int((~call & (y == 1)).sum()), int((~call & (y == 0)).sum())
    return {"sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "fnr": fn / max(tp + fn, 1), "fpr": fp / max(tn + fp, 1),
            "accuracy": (tp + tn) / max(len(y), 1), "n": len(y)}


def platt(y_train, s_train):
    """Logistic calibration map fitted on one set (soluble proteins), applied to others."""
    from sklearn.linear_model import LogisticRegression

    s = np.asarray(s_train, float)
    ok = np.isfinite(s)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(s[ok, None], np.asarray(y_train)[ok])
    return lambda x: m.predict_proba(np.asarray(x, float)[:, None])[:, 1]


def prior_shift(p, prev_from: float, prev_to: float):
    """Re-express probabilities calibrated at prevalence `prev_from` for a population with prevalence `prev_to`."""
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    logit = np.log(p / (1 - p)) + np.log(prev_to / (1 - prev_to)) - np.log(prev_from / (1 - prev_from))
    return 1 / (1 + np.exp(-logit))


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    k = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    df = pd.DataFrame({"k": k, "y": y, "p": p}).groupby("k").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    return df.reset_index(drop=True)


def ece(y, p, bins: int = 10) -> float:
    r = reliability(y, p, bins)
    return float((r.n * (r.pred - r.obs).abs()).sum() / r.n.sum())


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def cliffs_delta(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if not len(x) or not len(y):
        return np.nan
    u = stats.mannwhitneyu(x, y).statistic
    return float(2 * u / (len(x) * len(y)) - 1)


def mwu_p(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    return float(stats.mannwhitneyu(x, y).pvalue)


def bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values (NaN kept)."""
    p = pd.Series(p, dtype=float)
    ok = p.notna()
    q = p.copy()
    v = p[ok].to_numpy()
    order = np.argsort(v)
    ranked = v[order] * len(v) / (np.arange(len(v)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(v)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q
