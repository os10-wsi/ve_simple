"""Which membrane proteins does ThermoMPNN get wrong, and why?

Two rankings of MULTIPASS proteins:
* clinical: ThermoMPNN's per-protein ClinVar AUC (human; proteins with enough P and B variants);
* agreement: median per-protein Spearman rho between ThermoMPNN and the sequence-based variant-effect
  predictors over all substitutions (needs protein_correlations.parquet; works for yeast too).

For each protein the table collects identity (gene, name, family), the AUCs of every predictor, ThermoMPNN's
correlation with every other predictor and with residue context, and where its ClinVar variants sit.
`contrast` then asks which of these features separate the bottom proteins from the rest.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats as S
from .groups import DIRECTION, MEMBRANE

TARGET = "thermompnn"
PARTNERS = ["esm1v", "saprot", "alphamissense"]          # sequence/structure VEPs used for "agreement"
CONTEXT_LABEL = {"rsa": "relative SASA", "depth_mean": "burial depth", "plddt_model": "pLDDT", "in_tm": "in TM segment"}

# Families from UniProt "Belongs to the ... family" and, failing that, the gene symbol.
_FAMILY_TEXT = [
    ("G protein coupled receptor", "GPCR"), ("olfactory receptor", "GPCR"), ("taste receptor", "GPCR"),
    ("ABC transporter", "ABC transporter"), ("major facilitator", "Solute carrier"), ("solute carrier", "Solute carrier"),
    ("mitochondrial carrier", "Solute carrier"), ("amino acid-polyamine", "Solute carrier"), ("sodium:neurotransmitter", "Solute carrier"),
    ("potassium channel", "Ion channel"), ("sodium channel", "Ion channel"), ("calcium channel", "Ion channel"),
    ("chloride channel", "Ion channel"), ("cation channel", "Ion channel"), ("ion channel", "Ion channel"),
    ("ligand-gated", "Ion channel"), ("transient receptor", "Ion channel"), ("aquaporin", "Channel (aquaporin/connexin)"),
    ("connexin", "Channel (aquaporin/connexin)"), ("cation transport ATPase", "P-type ATPase"), ("P-type ATPase", "P-type ATPase"),
    ("claudin", "Tetraspanin / claudin"), ("tetraspanin", "Tetraspanin / claudin"), ("PMP-22", "Tetraspanin / claudin"),
    ("glycosyltransferase", "Enzyme"), ("cytochrome", "Enzyme"), ("acyltransferase", "Enzyme"), ("peptidase", "Enzyme"),
]
_FAMILY_GENE = [
    (r"^SLC", "Solute carrier"), (r"^ABC", "ABC transporter"), (r"^ATP\d", "P-type ATPase"),
    (r"^(KCN|SCN|CACN|HCN|TRP|CLCN|GABR|GRI[ANKD]|CHRN|P2RX|PIEZO|ANO\d|TMC\d|CNG|ASIC|PKD|TPCN|ORAI|GLR|ZACN|HTR3|LRRC8)", "Ion channel"),
    (r"^(OR\d|TAS\dR|GPR|ADR[AB]|DRD|HTR\d|CHRM|OPR|NPY\dR|CCR|CXCR|FZD|ADGR|GRM\d|LPAR|S1PR|P2RY|MC\dR|AVPR|EDNR|TACR)", "GPCR"),
    (r"^(CLDN|TSPAN|PMP22|CD\d)", "Tetraspanin / claudin"), (r"^(AQP|GJ[ABCD])", "Channel (aquaporin/connexin)"),
    (r"^TMEM", "TMEM (uncharacterised)"),
]


def _norm(x) -> str:
    return re.sub(r"[-_]", " ", str(x or "")).lower()


def family(gene: str | None, similarity: str | None, keywords: str | None = None) -> str:
    """Coarse family: UniProt family text first, then UniProt keywords, then the gene symbol."""
    for text in (_norm(similarity), _norm(keywords)):
        for key, fam in _FAMILY_TEXT:
            if _norm(key) in text:
                return fam
    g = (gene or "").upper()
    for pat, fam in _FAMILY_GENE:
        if re.match(pat, g):
            return fam
    return "Other"


def annotations(shape_dir: str | Path | None) -> pd.DataFrame:
    """UniProt similarity / keywords from the shape pipeline (for family assignment), if available."""
    p = Path(shape_dir) / "annotations.parquet" if shape_dir else None
    if p is None or not p.exists():
        return pd.DataFrame(columns=["accession", "similarity", "subcellular_location"])
    cols = [c for c in ("accession", "similarity", "keywords") if c in pd.read_parquet(p, columns=None).columns]
    a = pd.read_parquet(p, columns=cols)
    if "similarity" in a:
        a["similarity"] = a["similarity"].astype(str)
    return a


def labelled_features(lab: pd.DataFrame) -> pd.DataFrame:
    """Where each protein's ClinVar pathogenic and benign variants sit, and how ThermoMPNN scores them.
    `lab` is S.labels(...) output (y = 1 pathogenic, 0 benign)."""
    d = lab.copy()
    d["exposed"] = d["rsa"] >= 0.2
    d["tm"] = d["in_tm"].fillna(False).astype(bool)
    d["thermo"] = d[TARGET] * DIRECTION[TARGET]
    d["charge_change"] = d.get("abs_delta_charge", pd.Series(np.nan, index=d.index)) > 0
    d["to_pro_gly"] = d["mut"].isin(["P", "G"]) if "mut" in d else False
    out = []
    for acc, g in d.groupby("accession"):
        p, b = g[g.y == 1], g[g.y == 0]
        out.append({"accession": acc, "n_P": len(p), "n_B": len(b), "frac_pathogenic": len(p) / max(len(g), 1),
                    "P_in_TM": p.tm.mean() if len(p) else np.nan, "B_in_TM": b.tm.mean() if len(b) else np.nan,
                    "P_exposed": p.exposed.mean() if len(p) else np.nan, "B_exposed": b.exposed.mean() if len(b) else np.nan,
                    "P_median_rsa": p.rsa.median(), "B_median_rsa": b.rsa.median(),
                    "P_median_plddt": p.plddt_model.median(), "B_median_plddt": b.plddt_model.median(),
                    "P_charge_change": p.charge_change.mean() if len(p) else np.nan,
                    "P_to_pro_gly": p.to_pro_gly.mean() if len(p) else np.nan,
                    "thermo_P_median": p.thermo.median(), "thermo_B_median": b.thermo.median(),
                    "thermo_P_minus_B": p.thermo.median() - b.thermo.median()})
    return pd.DataFrame(out)


def protein_table(D: dict, lab: pd.DataFrame | None, pcorr: pd.DataFrame | None, min_per_class: int = 5,
                  classes: tuple = (MEMBRANE,)) -> pd.DataFrame:
    """One row per protein of `classes`: identity, family, per-predictor AUC, ThermoMPNN correlations,
    labelled-variant features."""
    prot = D["proteins"]
    t = prot[prot.in_ve_results & prot.class_annotation.isin(classes)].copy()
    keep = [c for c in ("accession", "gene", "protein_name", "length", "class_annotation", "n_tm_helical", "plddt_mean",
                        "frac_excluded", "n_pathogenic", "n_benign", "n_vus", "subcellular_location") if c in t]
    t = t[keep]
    ann = annotations(D.get("meta", {}).get("shape_results"))
    if len(ann):
        t = t.merge(ann[[c for c in ("accession", "similarity", "keywords") if c in ann]], on="accession", how="left")
    none = pd.Series([None] * len(t), index=t.index)
    t["family"] = [family(g, s, k) for g, s, k in zip(t.get("gene", none), t.get("similarity", none), t.get("keywords", none))]
    pauc = D.get("protein_auc", pd.DataFrame())
    if len(pauc):
        a = pauc[(pauc.n_pathogenic >= min_per_class) & (pauc.n_benign >= min_per_class)]
        w = a.pivot_table(index="accession", columns="predictor", values="auc")
        w.columns = [f"auc_{c}" for c in w.columns]
        t = t.merge(w, left_on="accession", right_index=True, how="left")
    if pcorr is not None and len(pcorr):
        c = pcorr[(pcorr.a == TARGET) | (pcorr.b == TARGET)].copy()
        c["other"] = np.where(c.a == TARGET, c.b, c.a)
        w = c.pivot_table(index="accession", columns="other", values="rho")
        w.columns = [f"rho_{x}" for x in w.columns]
        t = t.merge(w, left_on="accession", right_index=True, how="left")
        part = [f"rho_{p}" for p in PARTNERS if f"rho_{p}" in t]
        t["agreement"] = t[part].median(axis=1) if part else np.nan
    if lab is not None and len(lab):
        t = t.merge(labelled_features(lab), on="accession", how="left")
    return t


def bottom(t: pd.DataFrame, by: str, n: int = 100) -> pd.DataFrame:
    d = t.dropna(subset=[by]).sort_values(by)
    return d.head(n).assign(rank=np.arange(1, min(n, len(d)) + 1))


def contrast(t: pd.DataFrame, flag: pd.Series, cols: list[str]) -> pd.DataFrame:
    """Cliff's delta (bottom vs rest), medians and BH q for each numeric feature."""
    rows = []
    for c in cols:
        if c not in t or not pd.api.types.is_numeric_dtype(t[c]):
            continue
        x, y = t.loc[flag, c].dropna(), t.loc[~flag, c].dropna()
        if len(x) < 5 or len(y) < 5:
            continue
        rows.append({"feature": c, "median_bottom": x.median(), "median_rest": y.median(), "n_bottom": len(x),
                     "n_rest": len(y), "cliffs_delta": S.cliffs_delta(x, y), "p": S.mwu_p(x, y)})
    out = pd.DataFrame(rows)
    if len(out):
        out["q"] = S.bh(out["p"])
        out = out.sort_values("cliffs_delta", key=np.abs, ascending=False)
    return out


def family_counts(t: pd.DataFrame, flag: pd.Series) -> pd.DataFrame:
    a = t.loc[flag, "family"].value_counts().rename("bottom")
    b = t["family"].value_counts().rename("all")
    out = pd.concat([a, b], axis=1).fillna(0).astype(int)
    out["frac_bottom"] = out["bottom"] / max(out["bottom"].sum(), 1)
    out["frac_all"] = out["all"] / max(out["all"].sum(), 1)
    out["enrichment"] = out["frac_bottom"] / out["frac_all"].replace(0, np.nan)
    return out.sort_values("bottom", ascending=False)


FEATURE_LABEL = {
    "plddt_mean": "mean pLDDT", "frac_excluded": "fraction disordered / excluded", "length": "length",
    "n_tm_helical": "number of TM helices", "frac_pathogenic": "fraction pathogenic (ClinVar)",
    "P_in_TM": "pathogenic variants in TM", "B_in_TM": "benign variants in TM", "P_exposed": "pathogenic variants exposed",
    "B_exposed": "benign variants exposed", "P_median_rsa": "rel. SASA of pathogenic sites", "B_median_rsa": "rel. SASA of benign sites",
    "P_median_plddt": "pLDDT at pathogenic sites", "B_median_plddt": "pLDDT at benign sites",
    "P_charge_change": "pathogenic: charge change", "P_to_pro_gly": "pathogenic: to Pro/Gly",
    "thermo_P_minus_B": "ThermoMPNN ddG, pathogenic minus benign", "agreement": "agreement with sequence VEPs",
    "n_P": "pathogenic variants (n)", "n_B": "benign variants (n)",
}
