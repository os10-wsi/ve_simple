"""Stage 5 analyses on metrics.parquet: size dependence, size-matched membrane/soluble comparisons,
secondary-structure control, whole-protein view, prolateness, hydrophobicity vs depth, sensitivity,
TED globularity correlations, PCA/UMAP and findings.md.

Every function takes the metrics table and returns a DataFrame, so the notebook and the CLI share them.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

SIZE_BINS = [40, 80, 150, 300, np.inf]
SIZE_LABELS = ["40-80", "80-150", "150-300", "300+"]
NON_METRICS = {
    "accession", "taxid", "model_version", "metrics_version", "protein_class", "is_fragment", "protein_n_res",
    "unit", "mask_variant", "domain_id", "voxel_status", "unit_error", "class_annotation", "class_final",
    "belt_detected", "organism", "ted_id", "ted_chopping", "ted_shift", "ted_consensus", "ted_plddt",
    "ted_packing_density", "ted_norm_rg", "cath_label", "cath_class", "domain_n_tm_overlap", "domain_frac_tm",
    "domain_label", "n_res", "n_atoms", "mass_heavy", "size_bin", "group",
    "pa_x", "pa_y", "pa_z", "ca_pa_x", "ca_pa_y", "ca_pa_z",
}
GROUPS = ["MEMBRANE_DOMAIN", "SOLUBLE_DOMAIN_OF_MEMBRANE_PROTEIN", "SOLUBLE_DOMAIN"]
PAIRS = [("MEMBRANE_DOMAIN", "SOLUBLE_DOMAIN"), ("MEMBRANE_DOMAIN", "SOLUBLE_DOMAIN_OF_MEMBRANE_PROTEIN"),
         ("SOLUBLE_DOMAIN_OF_MEMBRANE_PROTEIN", "SOLUBLE_DOMAIN")]
HYDRO = ["h_depth_slope_kd", "inversion_index_kd", "belt_width_scan_A_kd", "belt_contrast_scan_kd",
         "hydrophobic_rg_ratio_kd", "h_depth_slope_atom", "inversion_index_atom", "h_mean_kd", "hmoment_kd"]
SHAPE = ["a_over_c", "kappa2", "prolateness", "asphericity", "acylindricity", "rg_ca_over_rmin_empirical",
         "solidity_ses", "sphericity_hull", "dmax_over_2rg"]


def metric_columns(m: pd.DataFrame, max_nan: float = 0.5) -> list[str]:
    cols = [c for c in m.columns if c not in NON_METRICS and pd.api.types.is_numeric_dtype(m[c])]
    return [c for c in cols if m[c].isna().mean() <= max_nan and m[c].nunique(dropna=True) > 1]


def domains(m: pd.DataFrame, variant: str = "default", class_col: str = "class_annotation") -> pd.DataFrame:
    d = m[(m.unit == "DOMAIN") & (m.mask_variant == variant)].copy()
    if class_col == "class_final":   # sensitivity: drop proteins the structural cross-check moved to AMBIGUOUS
        d = d[d.class_final == d.class_annotation]
    d["group"] = d["domain_label"]
    d["size_bin"] = pd.cut(d["n_res"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    return d


def chains(m: pd.DataFrame, unit: str = "CHAIN_TRIMMED", variant: str = "default") -> pd.DataFrame:
    c = m[(m.unit == unit) & (m.mask_variant == variant)].copy()
    c["group"] = c["class_annotation"]
    c["size_bin"] = pd.cut(c["n_res"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    return c


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) + 0.5 P(X = Y) (Mann-Whitney); Cliff's delta = 2 AUC - 1."""
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    return float(stats.mannwhitneyu(x, y, alternative="two-sided").statistic / (len(x) * len(y)))


# ----------------------------------------------------------------------------------- 1 size dependence
def size_dependence(m: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Spearman rho vs N and exponent nu (metric ~ N^nu) in the soluble reference population."""
    refs = {"DOMAIN": domains(m).query("group == 'SOLUBLE_DOMAIN'"),
            "CHAIN_TRIMMED": chains(m).query("group == 'SOLUBLE'"),
            "CHAIN_FULL": chains(m, "CHAIN_FULL").query("group == 'SOLUBLE'")}
    rows = []
    for unit, d in refs.items():
        for c in metrics:
            v = d[[c, "n_res"]].dropna()
            if len(v) < 10:
                continue
            rho = stats.spearmanr(v[c], v["n_res"]).statistic
            pos = v[v[c] > 0]
            nu = float(np.polyfit(np.log(pos["n_res"]), np.log(pos[c]), 1)[0]) if len(pos) >= 10 else np.nan
            rows.append({"unit": unit, "metric": c, "n": len(v), "spearman_rho_vs_N": rho, "nu": nu,
                         "size_dependent": abs(rho) > 0.3})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------- 2/3 comparisons
def compare(d: pd.DataFrame, metrics: list[str], a: str, b: str, strata=("taxid", "size_bin")) -> pd.DataFrame:
    """Per size bin (pooled over organisms) and a stratified estimate over organism x size bin."""
    rows = []
    for c in metrics:
        pooled_w, pooled_auc = 0.0, 0.0
        for sb in SIZE_LABELS:
            x = d[(d.group == a) & (d.size_bin == sb)][c].to_numpy(float)
            y = d[(d.group == b) & (d.size_bin == sb)][c].to_numpy(float)
            u = auc(x, y)
            rows.append({"metric": c, "a": a, "b": b, "size_bin": sb, "n_a": np.isfinite(x).sum(),
                         "n_b": np.isfinite(y).sum(), "auc": u, "cliffs_delta": 2 * u - 1 if np.isfinite(u) else np.nan})
        for _, g in d.groupby(list(strata), observed=True):
            x = g[g.group == a][c].to_numpy(float)
            y = g[g.group == b][c].to_numpy(float)
            x, y = x[np.isfinite(x)], y[np.isfinite(y)]
            if len(x) and len(y):
                w = len(x) * len(y)
                pooled_auc += w * auc(x, y)
                pooled_w += w
        u = pooled_auc / pooled_w if pooled_w else np.nan
        rows.append({"metric": c, "a": a, "b": b, "size_bin": "stratified", "n_a": int((d.group == a).sum()),
                     "n_b": int((d.group == b).sum()), "auc": u, "cliffs_delta": 2 * u - 1 if np.isfinite(u) else np.nan})
    return pd.DataFrame(rows)


def main_comparison(m, metrics, variant="default", class_col="class_annotation", organism=None):
    d = domains(m, variant, class_col)
    if organism is not None:
        d = d[d.taxid == organism]
    return pd.concat([compare(d, metrics, a, b) for a, b in PAIRS], ignore_index=True)


def cath1_control(m, metrics) -> pd.DataFrame:
    d = domains(m)
    d = d[d.cath_class == 1]
    out = compare(d, metrics, "MEMBRANE_DOMAIN", "SOLUBLE_DOMAIN")
    out["control"] = "CATH class 1 only"
    return out


def barrel_vs_cath2(m, metrics) -> pd.DataFrame:
    d = domains(m)
    barrel = d[(d.protein_class == "BETA_BARREL_MEMBRANE") & (d.group == "MEMBRANE_DOMAIN")].assign(group="BARREL")
    sol2 = d[(d.group == "SOLUBLE_DOMAIN") & (d.cath_class == 2)].assign(group="SOLUBLE_CATH2")
    return compare(pd.concat([barrel, sol2]), metrics, "BARREL", "SOLUBLE_CATH2", strata=("taxid",))


def surviving_metrics(main: pd.DataFrame, ctrl: pd.DataFrame, min_effect: float = 0.15) -> pd.DataFrame:
    """Metrics separating MEMBRANE from SOLUBLE domains with |AUC-0.5| >= min_effect both overall and
    within CATH class 1, in the same direction."""
    a = main[(main.size_bin == "stratified") & (main.a == "MEMBRANE_DOMAIN") & (main.b == "SOLUBLE_DOMAIN")]
    b = ctrl[ctrl.size_bin == "stratified"]
    j = a[["metric", "auc", "n_a", "n_b"]].merge(b[["metric", "auc", "n_a", "n_b"]], on="metric", suffixes=("_all", "_cath1"))
    j["effect_all"] = j["auc_all"] - 0.5
    j["effect_cath1"] = j["auc_cath1"] - 0.5
    j["survives_cath1_control"] = (j.effect_all.abs() >= min_effect) & (j.effect_cath1.abs() >= min_effect) & \
        (np.sign(j.effect_all) == np.sign(j.effect_cath1))
    return j.sort_values("effect_cath1", key=lambda s: -s.abs())


# ----------------------------------------------------------------------------------- 4 whole protein
def whole_protein(m, metrics) -> pd.DataFrame:
    c = chains(m)
    out = [compare(c, metrics, "MULTIPASS", "SOLUBLE").assign(comparison="CHAIN_TRIMMED MULTIPASS vs SOLUBLE")]
    core = m[(m.unit == "TM_CORE") & (m.protein_class == "MULTIPASS")].assign(group="TM_CORE")
    sol = domains(m).query("group == 'SOLUBLE_DOMAIN'")
    tc = pd.concat([core, sol])
    tc["size_bin"] = pd.cut(tc["n_res"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    out.append(compare(tc, metrics, "TM_CORE", "SOLUBLE_DOMAIN").assign(comparison="TM_CORE vs SOLUBLE_DOMAIN"))
    return pd.concat(out, ignore_index=True)


def elongation_location(m) -> pd.DataFrame:
    """Median elongation of the MULTIPASS chain, its TM core, its soluble domains and soluble proteins."""
    parts = {
        "MULTIPASS chain (trimmed)": chains(m).query("group == 'MULTIPASS'"),
        "MULTIPASS TM_CORE": m[(m.unit == "TM_CORE") & (m.protein_class == "MULTIPASS")],
        "MEMBRANE_DOMAIN": domains(m).query("group == 'MEMBRANE_DOMAIN'"),
        "SOLUBLE_DOMAIN_OF_MEMBRANE_PROTEIN": domains(m).query("group == 'SOLUBLE_DOMAIN_OF_MEMBRANE_PROTEIN'"),
        "SOLUBLE chain (trimmed)": chains(m).query("group == 'SOLUBLE'"),
        "SOLUBLE_DOMAIN": domains(m).query("group == 'SOLUBLE_DOMAIN'"),
    }
    rows = []
    for k, d in parts.items():
        rows.append({"set": k, "n": len(d), **{f"median_{c}": d[c].median() for c in ("n_res", "a_over_c", "kappa2", "prolateness")}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------- 5 prolateness
def prolateness_by_group(m) -> pd.DataFrame:
    sets = {g: domains(m).query("group == @g") for g in GROUPS}
    sets["TM_CORE (MULTIPASS)"] = m[(m.unit == "TM_CORE") & (m.protein_class == "MULTIPASS")]
    rows = []
    for k, d in sets.items():
        s = d["prolateness"].dropna()
        rows.append({"group": k, "n": len(s), "median_S": s.median(), "q25": s.quantile(0.25), "q75": s.quantile(0.75),
                     "frac_oblate": float((s < 0).mean()) if len(s) else np.nan,
                     "oblate_reference_fits": bool(len(s) and s.median() < 0 and (s < 0).mean() > 0.5)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------- 6 hydrophobicity
def hydrophobicity_by_group(m) -> pd.DataFrame:
    d = domains(m)
    return d.groupby("group")[[c for c in HYDRO if c in d.columns]].median().reset_index()


def structure_only_roc(m, metrics) -> pd.DataFrame:
    """Recover the UniProt membrane label from structure alone: per-metric AUC and a cross-validated
    logistic regression (protein-grouped folds) on hydrophobicity metrics and on all metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    rows = []
    sets = {"DOMAIN": domains(m)[lambda x: x.group.isin(["MEMBRANE_DOMAIN", "SOLUBLE_DOMAIN"])].assign(
                y=lambda x: (x.group == "MEMBRANE_DOMAIN").astype(int)),
            "CHAIN_TRIMMED": chains(m)[lambda x: x.group.isin(["MULTIPASS", "SOLUBLE"])].assign(
                y=lambda x: (x.group == "MULTIPASS").astype(int))}
    for unit, d in sets.items():
        if d.y.nunique() < 2:
            continue
        for c in metrics:
            a = auc(d.loc[d.y == 1, c].to_numpy(float), d.loc[d.y == 0, c].to_numpy(float))
            rows.append({"unit": unit, "model": c, "auc": a, "auc_oriented": max(a, 1 - a) if np.isfinite(a) else np.nan,
                         "n_pos": int(d.y.sum()), "n_neg": int((1 - d.y).sum())})
        for name, cols in (("logistic: hydrophobicity metrics", [c for c in HYDRO if c in metrics]),
                           ("logistic: all metrics", metrics)):
            n_splits = int(min(5, d.groupby("y").accession.nunique().min()))
            if n_splits < 2:
                continue
            model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
            p = cross_val_predict(model, d[cols], d.y, groups=d.accession, cv=GroupKFold(n_splits), method="predict_proba")[:, 1]
            a = roc_auc_score(d.y, p)
            rows.append({"unit": unit, "model": name, "auc": a, "auc_oriented": a, "n_pos": int(d.y.sum()),
                         "n_neg": int((1 - d.y).sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------- 7 sensitivity
def sensitivity(m, metrics, min_effect: float = 0.15) -> pd.DataFrame:
    base = main_comparison(m, metrics)
    base = base[(base.size_bin == "stratified") & (base.a == "MEMBRANE_DOMAIN") & (base.b == "SOLUBLE_DOMAIN")]
    out = base[["metric", "auc"]].rename(columns={"auc": "auc_default"})
    runs = {"plddt50": dict(variant="plddt50"), "plddt90": dict(variant="plddt90"),
            "class_final": dict(class_col="class_final")}
    for k, kw in runs.items():
        r = main_comparison(m, metrics, **kw)
        r = r[(r.size_bin == "stratified") & (r.a == "MEMBRANE_DOMAIN") & (r.b == "SOLUBLE_DOMAIN")]
        out = out.merge(r[["metric", "auc"]].rename(columns={"auc": f"auc_{k}"}), on="metric", how="left")
    # without the TED non-domain exclusion: whole-chain MULTIPASS vs SOLUBLE, default vs no_ted
    for v in ("default", "no_ted"):
        c = compare(chains(m, variant=v), metrics, "MULTIPASS", "SOLUBLE")
        c = c[c.size_bin == "stratified"]
        out = out.merge(c[["metric", "auc"]].rename(columns={"auc": f"auc_chain_{v}"}), on="metric", how="left")

    def call(a):
        return np.where(a - 0.5 >= min_effect, "membrane>soluble", np.where(0.5 - a >= min_effect, "membrane<soluble", "n.s."))
    out["call_default"] = call(out.auc_default)
    moved = []
    for _, r in out.iterrows():
        mv = [k for k in ("plddt50", "plddt90", "class_final") if call(r[f"auc_{k}"]) != r.call_default]
        if call(r.auc_chain_default) != call(r.auc_chain_no_ted):
            mv.append("no_ted (chain)")
        moved.append(", ".join(mv))
    out["conclusion_moves_with"] = moved
    return out


# ----------------------------------------------------------------------------------- 8 TED globularity
def ted_globularity_correlation(m, metrics) -> pd.DataFrame:
    d = domains(m)
    rows = []
    for t in ("ted_norm_rg", "ted_packing_density"):
        for c in metrics:
            v = d[[t, c]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(v) >= 10:
                rows.append({"ted_term": t, "metric": c, "n": len(v), "spearman": stats.spearmanr(v[t], v[c]).statistic})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.assign(_a=out["spearman"].abs()).sort_values(["ted_term", "_a"], ascending=[True, False]).drop(columns="_a")


# ----------------------------------------------------------------------------------- 9 PCA / UMAP
def embeddings(m, metrics) -> pd.DataFrame:
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    d = domains(m)
    X = StandardScaler().fit_transform(SimpleImputer().fit_transform(d[metrics]))
    pca = PCA(n_components=min(5, X.shape[1], len(X)))
    pcs = pca.fit_transform(X)
    out = d[["accession", "domain_id", "group", "cath_class", "taxid", "n_res"]].copy()
    out["organism"] = out["taxid"].map({9606: "human", 559292: "yeast"}).fillna(out["taxid"].astype(str))
    out["PC1"], out["PC2"] = pcs[:, 0], pcs[:, 1]
    out.attrs["explained"] = pca.explained_variance_ratio_.tolist()
    try:
        import umap

        emb = umap.UMAP(n_neighbors=min(15, len(X) - 1), random_state=0).fit_transform(X)
        out["UMAP1"], out["UMAP2"] = emb[:, 0], emb[:, 1]
    except Exception as e:  # noqa: BLE001
        log.warning("UMAP failed: %s", e)
    return out


def plot_embeddings(emb: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [("group", "group"), ("cath_class", "CATH class"), ("organism", "organism"), ("n_res", "N residues")]
    kinds = [k for k in ("PC", "UMAP") if f"{k}1" in emb]
    fig, axes = plt.subplots(len(kinds), 4, figsize=(18, 4.3 * len(kinds)), squeeze=False)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    for i, k in enumerate(kinds):
        for j, (col, title) in enumerate(cols):
            ax = axes[i, j]
            v = emb[col]
            if col == "n_res":
                sc = ax.scatter(emb[f"{k}1"], emb[f"{k}2"], c=np.log10(v), s=8, cmap="viridis")
                fig.colorbar(sc, ax=ax, label="log10 N")
            else:
                for n, lev in enumerate(sorted(v.dropna().unique(), key=str)):
                    s = emb[v == lev]
                    ax.scatter(s[f"{k}1"], s[f"{k}2"], s=8, color=palette[n % len(palette)], label=str(lev))
                ax.legend(fontsize=7, frameon=False)
            ax.set_title(f"{k}: {title}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_distributions(m: pd.DataFrame, path: Path, cols=("a_over_c", "kappa2", "prolateness", "inversion_index_kd",
                                                        "belt_contrast_scan_kd", "rg_ca_over_rmin_empirical")) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = domains(m)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colours = dict(zip(GROUPS, ["#eb6834", "#eda100", "#2a78d6"]))
    for ax, c in zip(axes.ravel(), cols):
        data = [d[d.group == g][c].dropna() for g in GROUPS]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, g in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(colours[g])
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, 4), ["membrane", "soluble of\nmembrane prot.", "soluble"], fontsize=8)
        ax.set_title(c, fontsize=10)
        ax.axhline(0, color="#c3c2b7", lw=0.8) if c in ("prolateness", "inversion_index_kd") else None
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------- 10 findings
def run_analysis(work: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(work / "metrics.parquet")
    metrics = metric_columns(m[(m.unit == "DOMAIN") & (m.mask_variant == "default")])
    res = {
        "size_dependence": size_dependence(m, metrics),
        "main_comparison": main_comparison(m, metrics),
        "main_comparison_human": main_comparison(m, metrics, organism=9606),
        "main_comparison_yeast": main_comparison(m, metrics, organism=559292),
        "cath1_control": cath1_control(m, metrics),
        "barrel_vs_cath2": barrel_vs_cath2(m, metrics),
        "whole_protein": whole_protein(m, metrics),
        "elongation_location": elongation_location(m),
        "prolateness": prolateness_by_group(m),
        "hydrophobicity_by_group": hydrophobicity_by_group(m),
        "structure_only_roc": structure_only_roc(m, metrics),
        "sensitivity": sensitivity(m, metrics),
        "ted_globularity": ted_globularity_correlation(m, metrics),
    }
    res["surviving_metrics"] = surviving_metrics(res["main_comparison"], res["cath1_control"])
    for k, v in res.items():
        v.to_csv(out / f"{k}.tsv", sep="\t", index=False)
    emb = embeddings(m, metrics)
    emb.to_csv(out / "embeddings.tsv", sep="\t", index=False)
    plot_embeddings(emb, out / "embeddings.png")
    plot_distributions(m, out / "distributions.png")
    write_findings(m, res, emb, work / "findings.md")
    log.info("analysis written to %s and %s", out, work / "findings.md")
    return res


def _md(df: pd.DataFrame, n: int = 15, floatfmt: str = ".3f") -> str:
    d = df.head(n).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:{floatfmt}}" if np.isfinite(v) else "")
    return d.to_markdown(index=False)


def write_findings(m, res, emb, path: Path) -> None:
    dom = domains(m)
    counts = dom.groupby(["taxid", "group"]).size().unstack(fill_value=0)
    sd = res["size_dependence"]
    sd_dom = sd[sd.unit == "DOMAIN"].assign(abs_rho=lambda x: x.spearman_rho_vs_N.abs()).sort_values("abs_rho")
    surv = res["surviving_metrics"]
    roc = res["structure_only_roc"].sort_values("auc_oriented", ascending=False)
    lines = [
        "# Findings", "",
        f"Proteins: {m.accession.nunique()}; unit rows: {len(m)}; domains (default mask): {len(dom)}.", "",
        "Domain groups by organism:", "", counts.to_markdown(), "",
        "## 1. Size invariance (Spearman |rho| vs N within SOLUBLE_DOMAIN; |rho| > 0.3 flagged)", "",
        "Most size-invariant metrics:", "", _md(sd_dom[["metric", "n", "spearman_rho_vs_N", "nu"]], 20), "",
        "Most size-dependent metrics:", "", _md(sd_dom.iloc[::-1][["metric", "n", "spearman_rho_vs_N", "nu"]], 10), "",
        "## 2-3. Membrane vs soluble domains after the CATH class 1 control", "",
        "Size- and organism-stratified AUC (P(membrane > soluble)); `survives` = |AUC-0.5| >= 0.15 overall and "
        "within CATH class 1, same direction.", "",
        _md(surv[["metric", "auc_all", "auc_cath1", "n_a_cath1", "n_b_cath1", "survives_cath1_control"]], 25), "",
        "## 5. Prolate vs oblate", "", _md(res["prolateness"]), "",
        "## 4. Where does elongation live?", "", _md(res["elongation_location"]), "",
        "## 6. Recovering the UniProt membrane label from structure alone", "",
        _md(roc[["unit", "model", "auc", "auc_oriented", "n_pos", "n_neg"]], 15), "",
        "## 7. Sensitivity (metrics whose membrane/soluble call changes)", "",
        _md(res["sensitivity"][res["sensitivity"].conclusion_moves_with != ""][["metric", "auc_default", "auc_plddt50",
            "auc_plddt90", "auc_class_final", "auc_chain_default", "auc_chain_no_ted", "conclusion_moves_with"]], 25), "",
        "## 8. What TED's globularity terms capture (top |Spearman|)", "",
        _md(res["ted_globularity"].assign(a=lambda x: x.spearman.abs()).sort_values("a", ascending=False)
            [["ted_term", "metric", "n", "spearman"]], 12), "",
        "## Caveats", "",
        "- Classes and domain labels come from UniProt annotation (`class_annotation`); `class_final` (after the "
        "structural belt cross-check) is used only as a sensitivity analysis, to avoid circularity with the "
        "structure-only ROC.",
        "- The 0.15 AUC-effect threshold is an exploratory choice. The belt-detector threshold (slab contrast >= 2.0 "
        "Kyte-Doolittle units) was chosen by inspecting the Stage 0 distributions, so the belt confusion matrix on "
        "Stage 0 is optimistic; re-check it on the full run.",
        "- TED assigns no domain to long coiled coils and other non-globular structure, so the default exclusion "
        "removes them entirely from CHAIN_TRIMMED; the no_ted variant keeps them.",
        "- The surface-based belt detector misses beta barrels: the water-filled pore is solvent-exposed surface and "
        "dilutes the hydrophobic band (VDAC1 contrast 0.96).",
        "- TED domains are from AFDB v4 models; domains whose sequence (md5) does not match the cached model "
        "within +-30 residues are dropped.",
        "- Small groups (e.g. beta barrels, CATH class 1 soluble domains in a given size bin) give unstable AUCs; "
        "check n columns.",
    ]
    if "explained" in emb.attrs:
        lines.insert(lines.index("## Caveats"), f"PCA explained variance (first 5 PCs): "
                     f"{', '.join(f'{v:.2f}' for v in emb.attrs['explained'])}. Figures: analysis/embeddings.png, "
                     "analysis/distributions.png.\n")
    path.write_text("\n".join(lines) + "\n")
