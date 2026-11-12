"""One function per question in the notebook. Each returns a tidy DataFrame (and never plots)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import stats as S
from .groups import (CLASS_ORDER, DIRECTION, LOCATIONS_MEMBRANE, LOCATIONS_OTHER, MEMBRANE, PREDICTOR_GROUP,
                     PUBLISHED_THRESHOLDS)

COMPARATORS = [c for c in CLASS_ORDER if c != MEMBRANE] + ["NON_MULTIPASS"]


def with_comparators(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Subsets per class plus the pooled 'all non-multipass' group."""
    out = {c: df[df.protein_class == c] for c in CLASS_ORDER}
    out["NON_MULTIPASS"] = df[df.protein_class.notna() & (df.protein_class != MEMBRANE)]
    return out


# ============================================================================ (i) clinical performance
def overview(proteins: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    p = proteins[proteins.in_ve_results]
    rows = []
    for c in CLASS_ORDER:
        pc = p[p.class_annotation == c]
        lc = lab[lab.protein_class == c]
        rows.append({"class": c, "proteins": len(pc), "proteins_with_P_and_B": int(((pc.n_pathogenic > 0) & (pc.n_benign > 0)).sum()),
                     "pathogenic": int(lc.clinvar_class.isin(["pathogenic", "likely_pathogenic"]).sum()),
                     "benign": int(lc.clinvar_class.isin(["benign", "likely_benign"]).sum()),
                     "vus": int((lc.clinvar_class == "vus").sum()),
                     "conflicting": int((lc.clinvar_class == "conflicting").sum())})
    return pd.DataFrame(rows)


def pooled_performance(lab: pd.DataFrame, predictors: list[str], n_boot: int = 300) -> pd.DataFrame:
    """AUC per class and predictor over all labelled variants (protein-bootstrap CI), plus MULTIPASS minus
    each comparator with a CI and BH-adjusted p across predictors."""
    rows = []
    sets = with_comparators(lab)
    for p in predictors:
        res = {c: S.pooled_auc(d, p, n_boot=n_boot, seed=hash((p, c)) % 2**32) for c, d in sets.items()}
        for c, r in res.items():
            row = {k: v for k, v in r.items() if k != "boot"}
            row["class"] = c
            if c != MEMBRANE:
                row.update({f"mp_minus_{k}": v for k, v in
                            S.delta(res[MEMBRANE]["boot"], r["boot"], res[MEMBRANE]["auc"], r["auc"]).items()})
            rows.append(row)
    out = pd.DataFrame(rows)
    out["predictor_group"] = out.predictor.map(PREDICTOR_GROUP)
    for c in COMPARATORS:
        m = out["class"] == c
        out.loc[m, "mp_minus_q"] = S.bh(out.loc[m, "mp_minus_p"]).to_numpy()
    return out


def roc_curves(lab: pd.DataFrame, predictor: str, classes: list[str]) -> dict[str, tuple]:
    from sklearn.metrics import roc_curve

    out = {}
    for c, d in with_comparators(lab).items():
        if c not in classes:
            continue
        d = d[["y", predictor]].dropna()
        if d.y.nunique() == 2:
            fpr, tpr, _ = roc_curve(d.y, d[predictor] * DIRECTION[predictor])
            out[c] = (fpr, tpr, S.auc(d.y, d[predictor] * DIRECTION[predictor]))
    return out


def per_protein_performance(pauc: pd.DataFrame, predictors: list[str], min_per_class: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-protein AUCs (from ve_simple) for proteins with >= min_per_class pathogenic AND benign variants.
    Returns (filtered per-protein table, class comparison: median per class, Cliff's delta and p vs MULTIPASS)."""
    d = pauc[(pauc.n_pathogenic >= min_per_class) & (pauc.n_benign >= min_per_class) & pauc.predictor.isin(predictors)]
    d = d.dropna(subset=["auc"])
    rows = []
    for p, g in d.groupby("predictor"):
        mp = g[g.protein_class == MEMBRANE].auc
        for c, sub in with_comparators(g).items():
            rows.append({"predictor": p, "class": c, "n_proteins": len(sub), "median_auc": sub.auc.median(),
                         "q25": sub.auc.quantile(0.25), "q75": sub.auc.quantile(0.75),
                         "frac_auc_below_0.7": float((sub.auc < 0.7).mean()) if len(sub) else np.nan,
                         "cliffs_delta_mp_vs": S.cliffs_delta(mp, sub.auc) if c != MEMBRANE else np.nan,
                         "p_mp_vs": S.mwu_p(mp, sub.auc) if c != MEMBRANE else np.nan})
    comp = pd.DataFrame(rows)
    if len(comp):
        comp["q_mp_vs"] = S.bh(comp["p_mp_vs"])
    return d, comp


def location_performance(lab: pd.DataFrame, predictors: list[str], n_boot: int = 200) -> pd.DataFrame:
    """AUC by residue location: inside MULTIPASS proteins (TM lipid-facing / TM buried / juxtamembrane /
    extramembrane) and inside SOLUBLE proteins (buried / exposed)."""
    rows = []
    for c, locs in ((MEMBRANE, LOCATIONS_MEMBRANE), ("SOLUBLE", LOCATIONS_OTHER)):
        dc = lab[lab.protein_class == c]
        for loc in locs:
            d = dc[dc.location == loc]
            for p in predictors:
                r = S.pooled_auc(d, p, n_boot=n_boot, seed=hash((c, loc, p)) % 2**32)
                r.pop("boot")
                rows.append({**r, "class": c, "location": loc})
    return pd.DataFrame(rows)


def add_strata(lab: pd.DataFrame) -> pd.DataFrame:
    d = lab.copy()
    d["plddt_bin"] = pd.cut(d["plddt_model"], [-1, 70, 90, 101], labels=["<70", "70-90", ">=90"]).astype(str)
    d["exposure_bin"] = np.where(d["rsa"] >= 0.2, "exposed", np.where(d["rsa"].notna(), "buried", "unknown"))
    frac = d.groupby("accession").y.transform("mean")
    d["pfrac_bin"] = pd.cut(frac, [-0.01, 1 / 3, 2 / 3, 1.01], labels=["low", "mid", "high"]).astype(str)
    return d


def matched_performance(lab: pd.DataFrame, predictors: list[str], n_boot: int = 100) -> pd.DataFrame:
    """Does the MULTIPASS-SOLUBLE gap survive matching?
    * raw: pooled AUC;
    * matched: within strata of residue pLDDT x burial x the protein's pathogenic fraction, averaged with the
      MULTIPASS stratum weights for both classes (same label mix and structural context);
    * within-protein: AUC computed inside each protein and averaged (weights n_P x n_B), removing any
      between-gene component."""
    d = add_strata(lab)
    strata = ["plddt_bin", "exposure_bin", "pfrac_bin"]
    mp, so = d[d.protein_class == MEMBRANE], d[d.protein_class == "SOLUBLE"]
    rows = []
    for p in predictors:
        for kind, fn in (("raw", lambda x, ref: S.auc(x.y, x[p] * DIRECTION[p])),
                         ("matched", lambda x, ref: S.stratified_auc(x, p, strata, ref)),
                         ("within-protein", lambda x, ref: S.stratified_auc(x, p, ["accession"], x))):
            a_mp, a_so = fn(mp, mp), fn(so, mp)
            b_mp = S.cluster_bootstrap(mp, lambda x: fn(x, mp), n_boot, seed=1) if n_boot else None
            b_so = S.cluster_bootstrap(so, lambda x: fn(x, mp), n_boot, seed=2) if n_boot else None
            ci = lambda b: (np.nanquantile(b, 0.025), np.nanquantile(b, 0.975)) if b is not None else (np.nan, np.nan)
            dl = S.delta(b_mp, b_so, a_mp, a_so)
            rows.append({"predictor": p, "estimate": kind, "auc_multipass": a_mp, "auc_multipass_ci": ci(b_mp),
                         "auc_soluble": a_so, "auc_soluble_ci": ci(b_so), **dl})
    out = pd.DataFrame(rows)
    out["q"] = S.bh(out["p"])
    return out


# ============================================================================ (ii) dynamic range and error
def percentile_ranks(sample: pd.DataFrame, df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Oriented scores of `df` expressed as percentiles of the proteome-wide distribution in `sample`
    (0 = most benign-looking, 100 = most pathogenic-looking), so predictors share one scale."""
    out = df.copy()
    for p in predictors:
        ref = np.sort((sample[p] * DIRECTION[p]).dropna().to_numpy())
        if len(ref):
            v = (df[p] * DIRECTION[p]).to_numpy(float)
            out[f"pct_{p}"] = np.where(np.isfinite(v), 100 * np.searchsorted(ref, v, side="right") / len(ref), np.nan)
    return out


def dynamic_range(pscores: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Per-protein dynamic range (5th-95th percentile of oriented scores over all substitutions), its split
    into between-position and within-position spread, and the comparison against MULTIPASS."""
    d = pscores[(pscores.region == "all") & pscores.predictor.isin(predictors)]
    rows = []
    for p, g in d.groupby("predictor"):
        mp = g[g.protein_class == MEMBRANE]
        for c, sub in with_comparators(g).items():
            row = {"predictor": p, "class": c, "n_proteins": len(sub)}
            for k in ("dynamic_range", "between_position_sd", "within_position_sd", "median"):
                row[f"median_{k}"] = sub[k].median()
                if c != MEMBRANE:
                    row[f"cliffs_{k}"] = S.cliffs_delta(mp[k], sub[k])
            rows.append(row)
    return pd.DataFrame(rows)


def tm_vs_rest_scores(pscores: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Within MULTIPASS proteins: are TM-segment substitutions scored as more severe, and over a narrower range?"""
    d = pscores[(pscores.protein_class == MEMBRANE) & pscores.region.isin(["TM", "Non-TM"]) & pscores.predictor.isin(predictors)]
    w = d.pivot_table(index=["accession", "predictor"], columns="region", values=["median", "dynamic_range"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.dropna().reset_index()
    rows = []
    for p, g in w.groupby("predictor"):
        rows.append({"predictor": p, "n_proteins": len(g),
                     "median_score_TM_minus_nonTM": float((g.median_TM - g["median_Non-TM"]).median()),
                     "p_median": float(stats.wilcoxon(g.median_TM, g["median_Non-TM"]).pvalue) if len(g) > 5 else np.nan,
                     "range_TM_over_nonTM": float((g.dynamic_range_TM / g["dynamic_range_Non-TM"]).median()),
                     "p_range": float(stats.wilcoxon(g.dynamic_range_TM, g["dynamic_range_Non-TM"]).pvalue) if len(g) > 5 else np.nan})
    return pd.DataFrame(rows)


def threshold_transfer(lab: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Thresholds tuned on SOLUBLE proteins (Youden J), or published ones, applied to every class:
    sensitivity, specificity, false-negative and false-positive rates."""
    so = lab[lab.protein_class == "SOLUBLE"]
    rows = []
    for p in predictors:
        s_so = so[p] * DIRECTION[p]
        ok = s_so.notna()
        if so.loc[ok, "y"].nunique() < 2:
            continue
        thr = S.youden_threshold(so.loc[ok, "y"], s_so[ok])
        cuts = [("Youden on soluble", thr)]
        if p in PUBLISHED_THRESHOLDS:
            cuts.append(("published", PUBLISHED_THRESHOLDS[p][1] * DIRECTION[p]))
        for name, t in cuts:
            for c, d in with_comparators(lab).items():
                s = d[p] * DIRECTION[p]
                rows.append({"predictor": p, "threshold": name, "threshold_value": t * DIRECTION[p], "class": c,
                             **S.error_rates(d.y, s, t)})
    return pd.DataFrame(rows)


def calibration(lab: pd.DataFrame, predictors: list[str], bins: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map each predictor to P(pathogenic) with a logistic fit on SOLUBLE variants, then check how well
    that map holds in every class: expected calibration error and Brier score, raw and after adjusting for
    each class's pathogenic fraction. Reliability curves are returned prevalence-adjusted."""
    so = lab[lab.protein_class == "SOLUBLE"]
    summ, curves = [], []
    for p in predictors:
        a = so[["y", p]].dropna()
        if a.y.nunique() < 2:
            continue
        f = S.platt(a.y, a[p] * DIRECTION[p])
        for c, d in with_comparators(lab).items():
            d = d[["y", p]].dropna()
            if len(d) < 30 or d.y.nunique() < 2:
                continue
            prob = f(d[p] * DIRECTION[p])
            # ClinVar's pathogenic fraction differs between classes (ascertainment, not biology). Shifting the
            # log-odds by the difference in prevalence removes that, leaving miscalibration in the score itself.
            adj = S.prior_shift(prob, a.y.mean(), d.y.mean())
            summ.append({"predictor": p, "class": c, "n": len(d), "fraction_pathogenic": float(d.y.mean()),
                         "ece": S.ece(d.y, prob, bins), "ece_prevalence_adjusted": S.ece(d.y, adj, bins),
                         "brier": S.brier(d.y, prob), "brier_prevalence_adjusted": S.brier(d.y, adj)})
            curves.append(S.reliability(d.y, adj, bins).assign(predictor=p, **{"class": c}))
    return pd.DataFrame(summ), (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame())


def alphamissense_zones(lab: pd.DataFrame) -> pd.DataFrame:
    """AlphaMissense calls with its published cut-offs per class: fraction of P/B/VUS variants in the
    'ambiguous' zone and fraction misclassified."""
    if "alphamissense" not in lab:
        return pd.DataFrame()
    lo, hi = PUBLISHED_THRESHOLDS["alphamissense"]
    rows = []
    for c, d in with_comparators(lab).items():
        d = d.dropna(subset=["alphamissense"])
        for lbl, sub in (("pathogenic", d[d.clinvar_class.isin(["pathogenic", "likely_pathogenic"])]),
                         ("benign", d[d.clinvar_class.isin(["benign", "likely_benign"])]),
                         ("vus", d[d.clinvar_class == "vus"])):
            if not len(sub):
                continue
            am = sub.alphamissense
            rows.append({"class": c, "clinvar": lbl, "n": len(sub), "called_benign": float((am < lo).mean()),
                         "ambiguous": float(((am >= lo) & (am <= hi)).mean()), "called_pathogenic": float((am > hi).mean())})
    return pd.DataFrame(rows)


def disagreement(df: pd.DataFrame, sample: pd.DataFrame, predictors: list[str], min_predictors: int = 3) -> pd.DataFrame:
    """Per variant: spread (SD) of the predictors' proteome percentiles. Higher = predictors disagree."""
    r = percentile_ranks(sample, df, predictors)
    cols = [f"pct_{p}" for p in predictors if f"pct_{p}" in r]
    n = r[cols].notna().sum(axis=1)
    r["disagreement_sd"] = r[cols].std(axis=1).where(n >= min_predictors)
    r["n_predictors"] = n
    return r


def correlation_by_class(df: pd.DataFrame, predictors: list[str], method: str = "spearman") -> dict[str, pd.DataFrame]:
    out = {}
    for c, d in with_comparators(df).items():
        o = pd.DataFrame({p: d[p] * DIRECTION[p] for p in predictors})
        if len(o.dropna(how="all")) > 30:
            out[c] = o.corr(method=method, min_periods=30)
    return out


def vus_burden(proteins: pd.DataFrame) -> pd.DataFrame:
    p = proteins[proteins.in_ve_results & (proteins.n_clinvar_missense > 0)].copy()
    p["vus_frac"] = p.n_vus / p.n_clinvar_missense
    p["resolved"] = p.n_pathogenic + p.n_benign
    rows = []
    for c, sub in with_comparators(p.rename(columns={"class_annotation": "protein_class"})).items():
        rows.append({"class": c, "proteins": len(sub), "variants": int(sub.n_clinvar_missense.sum()),
                     "vus_frac_pooled": float(sub.n_vus.sum() / max(sub.n_clinvar_missense.sum(), 1)),
                     "median_vus_frac_per_protein": float(sub.vus_frac.median()),
                     "vus_per_resolved": float(sub.n_vus.sum() / max(sub.resolved.sum(), 1))})
    return pd.DataFrame(rows)


def vus_reclassification(lab: pd.DataFrame, predictors: list[str], vep: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How predictors would call VUS, with SOLUBLE-tuned thresholds: per predictor fraction called
    pathogenic, and the agreement among the variant-effect predictors in `vep`.
    `lab` is the full ClinVar table (all classes, including VUS)."""
    so = S.labels(lab[lab.protein_class == "SOLUBLE"])          # thresholds from soluble P/B variants
    thr = {}
    for p in predictors:
        s = so[["y", p]].dropna()
        if s.y.nunique() == 2:
            thr[p] = S.youden_threshold(s.y, s[p] * DIRECTION[p])
    vus = lab[lab.clinvar_class == "vus"]
    calls = pd.DataFrame({p: np.where(vus[p].notna(), (vus[p] * DIRECTION[p] >= thr[p]).astype(float), np.nan)
                          for p in thr}, index=vus.index)
    per = []
    for c, idx in with_comparators(vus).items():
        for p in thr:
            v = calls.loc[idx.index, p].dropna()
            per.append({"class": c, "predictor": p, "n_vus": len(v), "called_pathogenic": float(v.mean()) if len(v) else np.nan})
    vcols = [p for p in vep if p in calls]
    agree = []
    if len(vcols) >= 2:
        k = calls[vcols].sum(axis=1, min_count=len(vcols))
        cat = np.select([k == len(vcols), k == 0, k.notna()], ["all pathogenic", "all benign", "split"], "missing")
        cat = pd.Series(cat, index=vus.index)
        for c, idx in with_comparators(vus).items():
            v = cat.loc[idx.index]
            v = v[v != "missing"]
            agree.append({"class": c, "n_vus": len(v), **{x: float((v == x).mean()) if len(v) else np.nan
                                                            for x in ("all pathogenic", "all benign", "split")}})
    return pd.DataFrame(per), pd.DataFrame(agree)


# ============================================================================ (iii) structure
STRUCT_METRICS = ["rg_ca_over_rmin_empirical", "a_over_c", "kappa2", "prolateness", "solidity_ses", "sphericity_hull",
                  "contact_density_ca", "sasa_scaled", "frac_helix", "frac_strand", "inversion_index_kd",
                  "h_depth_slope_kd", "hydrophobic_rg_ratio_kd", "belt_contrast_scan_kd", "cavity_frac",
                  "void_interstitial_frac", "plddt_mean"]
SIZE_BINS = [0, 80, 150, 300, 600, np.inf]


def structural_contrasts(shape: pd.DataFrame, unit: str = "CHAIN_TRIMMED", metrics: list[str] = STRUCT_METRICS,
                         taxid: int | None = None) -> pd.DataFrame:
    """Size-matched effect size (Cliff's delta, stratified by size bin) of MULTIPASS vs every other class."""
    d = shape[(shape.unit == unit)]
    if taxid is not None:
        d = d[d.taxid == taxid]
    d = d.assign(size_bin=pd.cut(d.n_res, SIZE_BINS))
    rows = []
    for m in [m for m in metrics if m in d]:
        mp = d[d.protein_class == MEMBRANE]
        for c, sub in with_comparators(d).items():
            if c == MEMBRANE:
                continue
            num, den = 0.0, 0.0
            for b, g in mp.groupby("size_bin", observed=True):
                o = sub[sub.size_bin == b]
                x, y = g[m].dropna(), o[m].dropna()
                if len(x) and len(y):
                    w = len(x) * len(y)
                    num += w * S.cliffs_delta(x, y)
                    den += w
            rows.append({"metric": m, "class": c, "unit": unit, "cliffs_delta_size_matched": num / den if den else np.nan,
                         "median_multipass": mp[m].median(), "median_other": sub[m].median(),
                         "n_multipass": int(mp[m].notna().sum()), "n_other": int(sub[m].notna().sum()),
                         "p": S.mwu_p(mp[m], sub[m])})
    out = pd.DataFrame(rows)
    if len(out):
        out["q"] = S.bh(out["p"])
    return out


def burial_by_location(residues: pd.DataFrame) -> pd.DataFrame:
    d = residues[residues.location != "Unknown"]
    return d.groupby(["protein_class", "location"]).agg(n=("rsa", "size"), median_rsa=("rsa", "median"),
                                                        median_depth=("depth_mean", "median"),
                                                        median_plddt=("plddt_model", "median"),
                                                        frac_helix=("dssp", lambda s: float((s == "H").mean()))).reset_index()


def auc_by_context(lab: pd.DataFrame, predictor: str, by: str, bins, labels_, n_boot: int = 100) -> pd.DataFrame:
    """Pooled AUC of MULTIPASS and SOLUBLE variants within bins of a residue property (burial depth, pLDDT)."""
    d = lab.assign(_bin=pd.cut(lab[by], bins, labels=labels_))
    rows = []
    for c in (MEMBRANE, "SOLUBLE"):
        for b in labels_:
            sub = d[(d.protein_class == c) & (d._bin == b)]
            r = S.pooled_auc(sub, predictor, n_boot=n_boot)
            r.pop("boot")
            rows.append({**r, "class": c, "bin": b})
    return pd.DataFrame(rows)


def auc_vs_structure(pauc: pd.DataFrame, shape: pd.DataFrame, predictor: str, metrics: list[str],
                     min_per_class: int = 5) -> pd.DataFrame:
    """Within each class, Spearman correlation between per-protein AUC and the protein's structural metrics."""
    a = pauc[(pauc.predictor == predictor) & (pauc.n_pathogenic >= min_per_class) & (pauc.n_benign >= min_per_class)]
    s = shape[shape.unit == "CHAIN_TRIMMED"][["accession"] + [m for m in metrics if m in shape]]
    j = a.merge(s, on="accession")
    rows = []
    for c in (MEMBRANE, "SOLUBLE"):
        g = j[j.protein_class == c]
        for m in metrics:
            v = g[["auc", m]].dropna() if m in g else pd.DataFrame()
            if len(v) >= 8:
                r = stats.spearmanr(v.auc, v[m])
                rows.append({"class": c, "metric": m, "n": len(v), "spearman": r.statistic, "p": r.pvalue})
    return pd.DataFrame(rows)
