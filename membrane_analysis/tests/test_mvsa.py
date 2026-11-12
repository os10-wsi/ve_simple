import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mvsa import stats as S                                       # noqa: E402
from mvsa.prepare import assign_location, summarise_protein        # noqa: E402

RNG = np.random.default_rng(0)


def toy(n_prot=40, n_var=30, shift=1.0, cls="SOLUBLE"):
    rows = []
    for i in range(n_prot):
        y = RNG.integers(0, 2, n_var)
        s = y * shift + RNG.normal(0, 1, n_var)
        rows.append(pd.DataFrame({"accession": f"{cls}{i}", "y": y, "alphamissense": s, "protein_class": cls}))
    return pd.concat(rows, ignore_index=True)


def test_labels_filters():
    df = pd.DataFrame({"clinvar_class": ["pathogenic", "likely_pathogenic", "benign", "likely_benign", "vus", "conflicting"],
                       "clinvar_review_status": ["criteria provided, multiple submitters", "no assertion criteria provided",
                                                 "reviewed by expert panel", "criteria provided, single submitter", "", ""]})
    assert list(S.labels(df).y) == [1, 1, 0, 0]
    assert list(S.labels(df, include_likely=False).y) == [1, 0]
    assert list(S.labels(df, min_stars=1).clinvar_class) == ["pathogenic", "benign", "likely_benign"]
    assert list(S.labels(df, min_stars=2).clinvar_class) == ["pathogenic", "benign"]


def test_pooled_auc_and_cluster_bootstrap():
    d = toy()
    r = S.pooled_auc(d, "alphamissense", n_boot=200)
    assert r["auc"] == pytest.approx(roc_auc_score(d.y, d.alphamissense))
    assert r["ci_low"] < r["auc"] < r["ci_high"]
    assert r["n_proteins"] == 40


def test_delta_detects_difference():
    a = S.pooled_auc(toy(shift=2.0, cls="A"), "alphamissense", n_boot=200, seed=1)
    b = S.pooled_auc(toy(shift=0.3, cls="B"), "alphamissense", n_boot=200, seed=2)
    dl = S.delta(a["boot"], b["boot"], a["auc"], b["auc"])
    assert dl["delta"] > 0.2 and dl["delta_low"] > 0 and dl["p"] < 0.01


def test_stratified_auc_removes_mix_effect():
    """Two strata with the same within-stratum separation: stratified AUC equals it even when the pooled
    AUC is inflated by stratum-level differences in the score."""
    rows = []
    for stratum, offset in (("a", 0.0), ("b", 5.0)):
        y = RNG.integers(0, 2, 2000)
        rows.append(pd.DataFrame({"accession": stratum, "st": stratum, "y": y,
                                  "alphamissense": y * 1.0 + RNG.normal(0, 1, 2000) + offset}))
    d = pd.concat(rows, ignore_index=True)
    d.loc[d.st == "b", "y"] = np.where(RNG.random((d.st == "b").sum()) < 0.9, 1, d.loc[d.st == "b", "y"])
    w = {k: g.y.sum() * (len(g) - g.y.sum()) for k, g in d.groupby("st")}          # n_pos x n_neg weights
    within = sum(w[k] * roc_auc_score(g.y, g.alphamissense) for k, g in d.groupby("st")) / sum(w.values())
    assert S.stratified_auc(d, "alphamissense", ["st"], d) == pytest.approx(within, abs=0.03)
    assert abs(roc_auc_score(d.y, d.alphamissense) - within) > 0.05          # pooling distorts it


def test_thresholds_calibration_and_effects():
    d = toy(shift=1.5)
    t = S.youden_threshold(d.y, d.alphamissense)
    e = S.error_rates(d.y, d.alphamissense, t)
    assert 0.6 < e["sensitivity"] < 1 and 0.6 < e["specificity"] < 1
    f = S.platt(d.y, d.alphamissense)
    p = f(d.alphamissense)
    assert S.ece(d.y, p) < 0.05 and 0 < S.brier(d.y, p) < 0.25
    assert S.cliffs_delta([3, 4, 5], [0, 1, 2]) == 1.0
    q = S.bh(pd.Series([0.01, 0.04, 0.03, np.nan]))
    assert q.iloc[0] == pytest.approx(0.03) and np.isnan(q.iloc[3])


def test_assign_location():
    cls = pd.Series(["MULTIPASS"] * 5 + ["SOLUBLE"] * 2)
    in_tm = pd.Series([True, True, False, False, False, False, False])
    juxta = pd.Series([False, False, True, False, False, False, False])
    rsa = pd.Series([0.5, 0.05, 0.3, 0.6, 0.1, 0.4, 0.1])
    assert list(assign_location(cls, in_tm, juxta, rsa)) == [
        "TM lipid-facing", "TM buried", "Juxtamembrane", "Extramembrane exposed", "Extramembrane buried",
        "Exposed", "Buried"]


def test_summarise_protein_dynamic_range():
    pos = np.repeat(np.arange(1, 21), 19)
    g = pd.DataFrame({"accession": "X", "position": pos, "alphamissense": pos / 20.0, "region": "Non-TM"})
    rows = {(r["region"], r["predictor"]): r for r in summarise_protein(g)}
    r = rows[("all", "alphamissense")]
    assert r["within_position_sd"] == pytest.approx(0.0)          # all substitutions at a position score alike
    assert r["between_position_sd"] > 0.25 and r["dynamic_range"] > 0.8
