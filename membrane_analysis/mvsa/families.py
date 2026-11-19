"""Compare predictors across membrane protein families.

The family representatives are run through ve_simple / protshape / mvsa as one dataset (e.g. data_families);
`membership` maps each accession to the family (or families) it was selected for, with its domain of life.
`protein_table` gives one row per protein x family with:
  dr_<p>       5-95% range of predictor p over all substitutions (oriented: higher = more damaging)
  med_<p>      median oriented score
  tm_<p>       median score of TM-segment substitutions minus non-TM substitutions
  rho_<a>__<b> per-protein Spearman rho between a and b over all substitutions (protein_correlations.parquet)
  shape metrics of the trimmed chain, and protein attributes (length, TM helices, pLDDT, disorder)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .analyses import STRUCT_METRICS

PAIRS = [("esm1v", "saprot"), ("esm1v", "thermompnn"), ("saprot", "thermompnn"), ("alphamissense", "thermompnn"),
         ("alphamissense", "esm1v"), ("esm1v", "blosum62"),
         ("thermompnn", "blosum62"), ("esm1v", "abs_delta_hydrophobicity"), ("thermompnn", "abs_delta_hydrophobicity"),
         ("esm1v", "solvent_exposure"), ("thermompnn", "solvent_exposure"), ("esm1v", "in_tm"), ("thermompnn", "in_tm"),
         ("esm1v", "plddt_model"), ("thermompnn", "plddt_model")]
SHAPE = ["frac_helix", "inversion_index_kd", "h_depth_slope_kd", "belt_contrast_scan_kd", "contact_density_ca",
         "cavity_frac", "a_over_c", "sphericity_hull", "plddt_mean"]
PROTEIN = ["length", "n_tm_helical", "plddt_mean", "frac_excluded"]


def pair_col(a: str, b: str) -> str:
    return f"rho_{a}__{b}"


def membership(path: str | Path) -> pd.DataFrame:
    m = pd.read_csv(path, sep="\t", dtype={"taxid": str})
    m["family"] = m["short"].astype(str)
    return m


def protein_table(D: dict, pcorr: pd.DataFrame | None, mem: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    prot = D["proteins"]
    base = prot[[c for c in ["accession", "gene", "protein_name", "class_annotation"] + PROTEIN if c in prot]]
    t = mem[["accession", "family", "pfam", "clan", "domain", "kingdom", "cluster_size", "reviewed"]].merge(base, on="accession", how="inner")
    ps = D.get("protein_scores", pd.DataFrame())
    if len(ps):
        ps = ps[ps.predictor.isin(predictors)]
        a = ps[ps.region == "all"]
        for col, src in (("dr", "dynamic_range"), ("med", "median")):
            w = a.pivot_table(index="accession", columns="predictor", values=src)
            w.columns = [f"{col}_{c}" for c in w.columns]
            t = t.merge(w, left_on="accession", right_index=True, how="left")
        tm = ps[ps.region.isin(["TM", "Non-TM"])].pivot_table(index=["accession", "predictor"], columns="region", values="median")
        if {"TM", "Non-TM"} <= set(tm.columns):
            w = (tm["TM"] - tm["Non-TM"]).unstack("predictor")
            w.columns = [f"tm_{c}" for c in w.columns]
            t = t.merge(w, left_on="accession", right_index=True, how="left")
    if pcorr is not None and len(pcorr):
        key = pcorr.a + "|" + pcorr.b
        for a_, b_ in PAIRS:
            sel = pcorr[(key == f"{a_}|{b_}") | (key == f"{b_}|{a_}")].set_index("accession").rho
            t[pair_col(a_, b_)] = t.accession.map(sel)
    su = D.get("shape_units", pd.DataFrame())
    if len(su):
        s = su[su.unit == "CHAIN_TRIMMED"]
        if "mask_variant" in s:
            s = s[s.mask_variant == "default"]
        s = s.drop_duplicates("accession").set_index("accession")
        for m in SHAPE:
            if m in s and m != "plddt_mean":
                t[m] = t.accession.map(s[m])
        if "plddt_mean" in s:
            t["plddt_trimmed"] = t.accession.map(s["plddt_mean"])
    return t


def family_summary(t: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in t]
    g = t.groupby("family")
    out = g[cols].median()
    out.insert(0, "n_proteins", g.size())
    out.insert(1, "frac_eukaryota", g["domain"].apply(lambda s: (s == "Eukaryota").mean()))
    out.insert(2, "frac_multipass", g["class_annotation"].apply(lambda s: (s == "MULTIPASS").mean()))
    return out


def drivers(t: pd.DataFrame, target: str, features: list[str], min_n: int = 20) -> pd.DataFrame:
    """Spearman rho of `target` with each feature: over all proteins, and the median of within-family rhos
    (which removes differences between families)."""
    rows = []
    for f in [f for f in features if f in t and f != target]:
        d = t[[target, f, "family"]].dropna()
        if len(d) < min_n:
            continue
        pooled = stats.spearmanr(d[target], d[f]).statistic
        within = [stats.spearmanr(g[target], g[f]).statistic for _, g in d.groupby("family") if len(g) >= min_n and g[f].nunique() > 2]
        rows.append({"feature": f, "rho_pooled": pooled, "rho_within_median": np.nanmedian(within) if within else np.nan,
                     "n_families": len(within), "n": len(d)})
    out = pd.DataFrame(rows, columns=["feature", "rho_pooled", "rho_within_median", "n_families", "n"])
    return out.sort_values("rho_within_median", key=np.abs, ascending=False)


def main_pair(t: pd.DataFrame, min_n: int = 20) -> str | None:
    """The sequence-vs-stability pair used for 'agreement': ESM-1v-ThermoMPNN, or the next one with data."""
    for a, b in (("esm1v", "thermompnn"), ("saprot", "thermompnn"), ("alphamissense", "thermompnn")):
        c = pair_col(a, b)
        if c in t and t[c].notna().sum() >= min_n:
            return c
    return None


def domain_effect(t: pd.DataFrame, col: str, min_n: int = 15) -> pd.DataFrame:
    """Per family: median of `col` in eukaryotic vs bacterial representatives."""
    rows = []
    for fam, g in t.groupby("family"):
        e, b = g[g.domain == "Eukaryota"][col].dropna(), g[g.domain == "Bacteria"][col].dropna()
        if len(e) >= min_n and len(b) >= min_n:
            rows.append({"family": fam, "eukaryota": e.median(), "bacteria": b.median(), "n_euk": len(e), "n_bac": len(b),
                         "p": stats.mannwhitneyu(e, b).pvalue})
    return pd.DataFrame(rows)
