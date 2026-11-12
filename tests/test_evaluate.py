import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ve_simple.evaluate import auc_for_protein, delong_auc_variance, roc_points, wide_auc_table


def _variants(n=60, seed=1):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n // 2), np.zeros(n // 2)]
    good = y * 2 + rng.normal(size=n)           # higher = pathogenic
    inverted = -good                             # predictor oriented the other way (esm1v-like)
    df = pd.DataFrame({
        "accession": "P1", "position": np.arange(1, n + 1), "wt": "A", "mut": "G",
        "alphamissense": good, "esm1v": inverted, "foldx": np.nan,
        "clinvar_label": np.where(y == 1, "pathogenic", "benign"),
    })
    df.loc[0:4, "alphamissense"] = np.nan   # a few unscored variants
    return df, y, good


def test_delong_matches_sklearn():
    df, y, good = _variants()
    auc, var = delong_auc_variance(good[y == 1], good[y == 0])
    assert np.isclose(auc, roc_auc_score(y, good))
    assert var > 0


def test_auc_table_orientation_and_counts():
    df, y, good = _variants()
    out = auc_for_protein(df, "P1", n_boot=200).set_index("predictor")
    assert np.isclose(out.loc["alphamissense", "auc"], roc_auc_score(y[5:], good[5:]))
    # inverted predictor with direction -1 gives the same AUC as the well-oriented one
    assert np.isclose(out.loc["esm1v", "auc"], roc_auc_score(y, good))
    assert out.loc["esm1v", "ci_low"] <= out.loc["esm1v", "auc"] <= out.loc["esm1v", "ci_high"]
    assert out.loc["esm1v", "auc_se"] > 0
    assert out.loc["alphamissense", "n_scored"] == 55
    assert np.isnan(out.loc["foldx", "auc"]) and out.loc["foldx", "n_scored"] == 0


def test_single_class_gives_nan():
    df, *_ = _variants()
    df["clinvar_label"] = "pathogenic"
    out = auc_for_protein(df, "P1", n_boot=10)
    assert out["auc"].isna().all()


def test_roc_points_and_wide():
    df, y, good = _variants()
    fpr, tpr, auc = roc_points(df, "esm1v")
    assert fpr[0] == 0 and tpr[-1] == 1 and np.isclose(auc, roc_auc_score(y, good))
    long = auc_for_protein(df, "P1", n_boot=10)
    wide = wide_auc_table(long, pd.DataFrame([{"accession": "P1", "gene": "G1", "length": 60}]))
    assert len(wide) == 1 and "esm1v_auc" in wide.columns and "esm1v_ci_high" in wide.columns


def test_delong_se_matches_bootstrap_sd():
    rng = np.random.default_rng(3)
    pos = rng.normal(1.0, 1, 150)
    neg = rng.normal(0.0, 1, 100)
    auc, var = delong_auc_variance(pos, neg)
    boots = []
    for _ in range(400):
        p = rng.choice(pos, len(pos)); q = rng.choice(neg, len(neg))
        boots.append(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(q))], np.r_[p, q]))
    assert 0.5 * np.std(boots) < np.sqrt(var) < 1.5 * np.std(boots)
    assert np.sqrt(var) < 0.06


def test_vectorised_bootstrap_matches_definition():
    from ve_simple.evaluate import bootstrap_ci

    rng = np.random.default_rng(5)
    pos = np.round(rng.normal(1.0, 1, 80), 1)      # rounding creates ties
    neg = np.round(rng.normal(0.0, 1, 40), 1)
    lo, hi = bootstrap_ci(pos, neg, n_boot=2000, seed=1)
    auc, var = delong_auc_variance(pos, neg)
    assert lo < auc < hi
    assert (hi - lo) == pytest.approx(2 * 1.96 * np.sqrt(var), rel=0.25)
    # a single resample scored by ranks equals sklearn's AUC
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    from scipy import stats
    r = stats.rankdata(np.r_[pos, neg])
    assert (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)) == pytest.approx(
        roc_auc_score(y, np.r_[pos, neg]))
