"""Takeaway text and the summary table, generated from the computed results (no hand-written claims)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .groups import CLASS_LABEL, MEMBRANE, PREDICTOR_LABEL


def _f(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _q(q):
    return "n/a" if not np.isfinite(q) else ("< 0.001" if q < 0.001 else f"{q:.3f}")


def performance_takeaway(pooled: pd.DataFrame, comparator: str = "SOLUBLE", alpha: float = 0.05) -> str:
    d = pooled[pooled["class"] == comparator].dropna(subset=["mp_minus_delta"])
    if d.empty:
        return "_Not enough labelled variants to compare._"
    worse = d[(d.mp_minus_q < alpha) & (d.mp_minus_delta < 0)]
    better = d[(d.mp_minus_q < alpha) & (d.mp_minus_delta > 0)]
    mp = pooled[pooled["class"] == MEMBRANE].set_index("predictor")
    lines = [f"**Against {CLASS_LABEL.get(comparator, comparator).lower()} proteins:** "
             f"{len(worse)} of {len(d)} predictors are significantly worse on multipass membrane proteins "
             f"(BH q < {alpha}), {len(better)} significantly better.", ""]
    for r in d.sort_values("mp_minus_delta").itertuples():
        m = mp.loc[r.predictor]
        lines.append(f"- {PREDICTOR_LABEL.get(r.predictor, r.predictor)}: AUC {_f(m.auc)} ({_f(m.ci_low)}-{_f(m.ci_high)}) "
                     f"vs {_f(r.auc)} ({_f(r.ci_low)}-{_f(r.ci_high)}); difference {_f(r.mp_minus_delta)} "
                     f"({_f(r.mp_minus_delta_low)} to {_f(r.mp_minus_delta_high)}), q {_q(r.mp_minus_q)}")
    return "\n".join(lines)


def matched_takeaway(matched: pd.DataFrame) -> str:
    if matched.empty:
        return "_No matched comparison available._"
    lines = ["Multipass minus soluble AUC, before and after matching (95% CI):", ""]
    for p, g in matched.groupby("predictor", sort=False):
        g = g.set_index("estimate")
        parts = [f"{k} {_f(g.at[k, 'delta'])} ({_f(g.at[k, 'delta_low'])} to {_f(g.at[k, 'delta_high'])})"
                 for k in ("raw", "matched", "within-protein") if k in g.index]
        lines.append(f"- {PREDICTOR_LABEL.get(p, p)}: " + "; ".join(parts))
    return "\n".join(lines)


def calibration_takeaway(calib: pd.DataFrame) -> str:
    if calib.empty:
        return "_No calibration results._"
    w = calib.pivot_table(index="predictor", columns="class", values=["ece", "ece_prevalence_adjusted", "fraction_pathogenic"])
    if ("ece", MEMBRANE) not in w or ("ece", "SOLUBLE") not in w:
        return "_Calibration needs both classes._"
    w = w.dropna(subset=[("ece_prevalence_adjusted", MEMBRANE), ("ece_prevalence_adjusted", "SOLUBLE")])
    adj_worse = (w[("ece_prevalence_adjusted", MEMBRANE)] > w[("ece_prevalence_adjusted", "SOLUBLE")]).sum()
    fp = w[("fraction_pathogenic", MEMBRANE)].median(), w[("fraction_pathogenic", "SOLUBLE")].median()
    lines = [f"Labelled ClinVar variants are {_f(fp[0])} pathogenic in multipass vs {_f(fp[1])} in soluble proteins, so a "
             f"probability map fitted on soluble proteins is off on multipass proteins for that reason alone. After "
             f"adjusting for each class's pathogenic fraction, the calibration error is still higher on multipass proteins "
             f"for {adj_worse} of {len(w)} predictors.", ""]
    for p, r in w.iterrows():
        lines.append(f"- {PREDICTOR_LABEL.get(p, p)}: adjusted ECE {_f(r[('ece_prevalence_adjusted', MEMBRANE)], 3)} multipass vs "
                     f"{_f(r[('ece_prevalence_adjusted', 'SOLUBLE')], 3)} soluble (unadjusted {_f(r[('ece', MEMBRANE)], 3)} vs "
                     f"{_f(r[('ece', 'SOLUBLE')], 3)})")
    return "\n".join(lines)


def threshold_takeaway(thr: pd.DataFrame) -> str:
    d = thr[(thr.threshold == "Youden on soluble") & thr["class"].isin([MEMBRANE, "SOLUBLE"])]
    if d.empty:
        return "_No threshold results._"
    w = d.pivot_table(index="predictor", columns="class", values=["fnr", "fpr"])
    lines = ["Thresholds tuned on soluble proteins, applied unchanged to multipass proteins:", ""]
    for p, r in w.iterrows():
        lines.append(f"- {PREDICTOR_LABEL.get(p, p)}: false-negative rate {_f(r[('fnr', MEMBRANE)])} vs "
                     f"{_f(r[('fnr', 'SOLUBLE')])}; false-positive rate {_f(r[('fpr', MEMBRANE)])} vs {_f(r[('fpr', 'SOLUBLE')])}")
    return "\n".join(lines)


def structure_takeaway(contrasts: pd.DataFrame, comparator: str = "SOLUBLE", top: int = 6) -> str:
    d = contrasts[contrasts["class"] == comparator].dropna(subset=["cliffs_delta_size_matched"])
    if d.empty:
        return "_No structural contrasts._"
    d = d.reindex(d.cliffs_delta_size_matched.abs().sort_values(ascending=False).index).head(top)
    lines = [f"Largest size-matched differences, multipass vs {CLASS_LABEL.get(comparator, comparator).lower()} "
             f"(Cliff's delta; +1 = always higher in multipass):", ""]
    for r in d.itertuples():
        lines.append(f"- {r.metric}: delta {_f(r.cliffs_delta_size_matched)} (median {_f(r.median_multipass)} vs "
                     f"{_f(r.median_other)}; q {_q(r.q)})")
    return "\n".join(lines)


def summary_table(pooled, matched, thr, calib, dyn, contrasts) -> pd.DataFrame:
    """One row per headline contrast: multipass vs soluble, with estimate and CI where available."""
    rows = []
    for r in pooled[pooled["class"] == "SOLUBLE"].itertuples():
        rows.append({"section": "(i) performance", "measure": "pooled AUC difference (MP - SOL)", "item": r.predictor,
                     "estimate": r.mp_minus_delta, "low": r.mp_minus_delta_low, "high": r.mp_minus_delta_high, "q": r.mp_minus_q})
    for r in matched[matched.estimate != "raw"].itertuples():
        rows.append({"section": "(i) performance", "measure": f"{r.estimate} AUC difference (MP - SOL)", "item": r.predictor,
                     "estimate": r.delta, "low": r.delta_low, "high": r.delta_high, "q": r.q})
    for r in thr[(thr.threshold == "Youden on soluble") & (thr["class"] == MEMBRANE)].itertuples():
        so = thr[(thr.threshold == "Youden on soluble") & (thr["class"] == "SOLUBLE") & (thr.predictor == r.predictor)]
        if len(so):
            rows.append({"section": "(ii) error", "measure": "false-negative rate difference (MP - SOL)",
                         "item": r.predictor, "estimate": r.fnr - so.fnr.iat[0], "low": np.nan, "high": np.nan, "q": np.nan})
    if len(calib):
        w = calib.pivot_table(index="predictor", columns="class", values="ece_prevalence_adjusted")
        for p, r in w.iterrows():
            if MEMBRANE in r and "SOLUBLE" in r:
                rows.append({"section": "(ii) error", "measure": "prevalence-adjusted calibration error difference (MP - SOL)", "item": p,
                             "estimate": r[MEMBRANE] - r["SOLUBLE"], "low": np.nan, "high": np.nan, "q": np.nan})
    for r in dyn[dyn["class"] == "SOLUBLE"].itertuples():
        rows.append({"section": "(ii) dynamic range", "measure": "per-protein dynamic range (Cliff's delta MP vs SOL)",
                     "item": r.predictor, "estimate": r.cliffs_dynamic_range, "low": np.nan, "high": np.nan, "q": np.nan})
    for r in contrasts[contrasts["class"] == "SOLUBLE"].itertuples():
        rows.append({"section": "(iii) structure", "measure": f"{r.unit} size-matched Cliff's delta (MP vs SOL)",
                     "item": r.metric, "estimate": r.cliffs_delta_size_matched, "low": np.nan, "high": np.nan, "q": r.q})
    return pd.DataFrame(rows)
