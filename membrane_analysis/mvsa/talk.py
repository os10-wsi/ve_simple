"""Slide figures: one message per figure, large type, no grid, one accent colour (multipass), greys for the rest.

Every function returns a matplotlib Figure; `headline` puts the take-away above the axes and `save`
writes PNG (200 dpi) and PDF into the figure directory.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .groups import CLASS_LABEL, MEMBRANE, PREDICTOR_LABEL

MP, SOL, OTHER, LIGHT = "#b8452a", "#34557a", "#8a8a8a", "#d9d9d9"
INK, SUB = "#1f1f1f", "#6b6b6b"
HUMAN_C, YEAST_C = "#1f1f1f", "#c49a3a"
DIVERGING = LinearSegmentedColormap.from_list("slate_brick", [SOL, "#f4f4f4", MP])
SEQUENTIAL = LinearSegmentedColormap.from_list("white_ink", ["#ffffff", "#3a3a3a"])
WIDE, TALL = (11, 5.6), (11, 6.2)
FIG_DIR = Path("figures_talk")

SHORT = {"alphamissense": "AlphaMissense", "esm1v": "ESM-1v", "saprot": "SaProt", "thermompnn": "ThermoMPNN",
         "foldx": "FoldX", "blosum62": "BLOSUM62", "abs_delta_hydrophobicity": "|Δ hydrophobicity|",
         "abs_delta_volume": "|Δ volume|", "abs_delta_charge": "|Δ charge|", "solvent_exposure": "Burial",
         "rsa": "Rel. SASA", "depth_mean": "Burial depth", "plddt_model": "pLDDT", "in_tm": "In TM segment"}


def name(x: str) -> str:
    return SHORT.get(x, PREDICTOR_LABEL.get(x, CLASS_LABEL.get(x, x)))


def flabel(f: str, labels: dict | None = None) -> str:
    """Readable label for a feature column of the ThermoMPNN tables."""
    if labels and f in labels:
        return labels[f]
    if f.startswith("rho_"):
        return "ρ with " + name(f[4:])
    if f.startswith("auc_"):
        return "AUC of " + name(f[4:])
    return name(f)


def fnum(v: float) -> str:
    if not np.isfinite(v):
        return "–"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}"
    return f"{v:.2f}"


def style(fig_dir: str | Path = "figures_talk") -> None:
    global FIG_DIR
    FIG_DIR = Path(fig_dir)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 14, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 14,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.0, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.grid": False,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 4, "ytick.major.size": 0,
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.facecolor": "white", "figure.facecolor": "white",
        "pdf.fonttype": 42, "lines.solid_capstyle": "round",
    })


def headline(fig, title: str, sub: str | None = None) -> None:
    fig.text(0.01, 1.0, title, ha="left", va="bottom", fontsize=20, fontweight="bold", color=INK)
    if sub:
        fig.text(0.01, 0.985, sub, ha="left", va="top", fontsize=13.5, color=SUB)
    fig.subplots_adjust(top=0.86 if sub else 0.93)


def save(fig, name_: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name_}.{ext}", bbox_inches="tight", pad_inches=0.25)


def _legend_dots(ax, items, loc="lower right"):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=m, ls="", color=c, mfc=f, mec=c, ms=10, mew=1.6, label=l) for l, c, m, f in items]
    ax.legend(handles=h, loc=loc, handletextpad=0.3)


# ------------------------------------------------------------------------------------------- clinical
def auc_dumbbell(pooled: pd.DataFrame, predictors: list[str], comparator: str = "SOLUBLE"):
    d = pooled[pooled.predictor.isin(predictors)]
    mp, so = d[d["class"] == MEMBRANE].set_index("predictor"), d[d["class"] == comparator].set_index("predictor")
    order = mp.auc.sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=WIDE)
    for i, p in enumerate(order):
        a, b = mp.loc[p], so.loc[p]
        ax.plot([a.auc, b.auc], [i, i], color=LIGHT, lw=6, zorder=1)
        ax.plot([b.ci_low, b.ci_high], [i - 0.12] * 2, color=SOL, lw=1.5)
        ax.plot([a.ci_low, a.ci_high], [i + 0.12] * 2, color=MP, lw=1.5)
        ax.scatter(b.auc, i - 0.12, s=90, color="white", edgecolor=SOL, lw=2, zorder=3)
        ax.scatter(a.auc, i + 0.12, s=90, color=MP, zorder=3)
        dlt = a.auc - b.auc
        ax.text(1.0, i, f"{dlt:+.3f}", va="center", ha="left", fontsize=12, color=MP if abs(dlt) >= 0.02 else SUB)
    ax.set_yticks(range(len(order)), [name(p) for p in order])
    ax.axvline(0.5, color=LIGHT, lw=1, zorder=0)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC, ClinVar pathogenic vs benign (95% CI, proteins resampled)")
    ax.text(1.0, len(order) - 0.35, "Δ", ha="left", fontsize=12, color=SUB)
    _legend_dots(ax, [("Multipass membrane", MP, "o", MP), (CLASS_LABEL[comparator], SOL, "o", "white")])
    return fig


def location_dots(loc: pd.DataFrame, predictors: list[str]):
    """Multipass AUC by location, each next to the matching soluble reference (exposed or buried)."""
    rows = [("TM lipid-facing", "Exposed"), ("TM buried", "Buried"), ("Juxtamembrane", None),
            ("Extramembrane exposed", "Exposed"), ("Extramembrane buried", "Buried")]
    preds = [p for p in predictors if p in set(loc.predictor)]
    fig, axes = plt.subplots(1, len(preds), figsize=(3.0 * len(preds) + 2.5, 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, preds):
        d = loc[loc.predictor == p]
        for i, (l, ref) in enumerate(rows):
            m = d[(d["class"] == MEMBRANE) & (d.location == l)]
            if ref:
                s = d[(d["class"] == "SOLUBLE") & (d.location == ref)]
                if len(s):
                    ax.scatter(s.auc, [i], s=80, color="white", edgecolor=SOL, lw=2, zorder=3)
                    if len(m):
                        ax.plot([s.auc.iat[0], m.auc.iat[0]], [i, i], color=LIGHT, lw=5, zorder=1)
            if len(m):
                ax.plot([m.ci_low.iat[0], m.ci_high.iat[0]], [i, i], color=MP, lw=1.5, zorder=2)
                ax.scatter(m.auc, [i], s=90, color=MP, zorder=4)
        ax.set_title(name(p), loc="left")
        ax.set_xlabel("AUC")
        lo = np.nanmin(d[["ci_low", "auc"]].to_numpy()) - 0.02
        ax.set_xlim(max(0.5, lo), min(1.0, np.nanmax(d[["ci_high", "auc"]].to_numpy()) + 0.02))
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    axes[0].invert_yaxis()
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=MP, ms=10, label="Multipass, this location"),
         Line2D([], [], marker="o", ls="", color=SOL, mfc="white", mew=1.8, ms=10, label="Soluble, same exposure")]
    fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.06))
    return fig


def frac_poor(pp: pd.DataFrame, predictors: list[str], cutoff: float = 0.7):
    """Fraction of proteins whose own AUC is below `cutoff`, multipass vs soluble."""
    rows = []
    for p in predictors:
        for c in (MEMBRANE, "SOLUBLE"):
            a = pp[(pp.predictor == p) & (pp.protein_class == c)].auc
            if len(a):
                rows.append({"predictor": p, "class": c, "frac": float((a < cutoff).mean()), "n": len(a)})
    d = pd.DataFrame(rows)
    order = d[d["class"] == MEMBRANE].sort_values("frac").predictor.tolist()
    fig, ax = plt.subplots(figsize=WIDE)
    for i, p in enumerate(order):
        for k, (c, col) in enumerate(((MEMBRANE, MP), ("SOLUBLE", SOL))):
            v = d[(d.predictor == p) & (d["class"] == c)]
            if len(v):
                ax.barh(i + (0.2 if k == 0 else -0.2), v.frac.iat[0], height=0.38, color=col if k == 0 else "white",
                        edgecolor=col, lw=1.6)
                ax.text(v.frac.iat[0] + 0.01, i + (0.2 if k == 0 else -0.2), f"{v.frac.iat[0]:.0%}", va="center",
                        fontsize=11.5, color=col)
    ax.set_yticks(range(len(order)), [name(p) for p in order])
    ax.set_xlabel(f"Fraction of proteins with per-protein AUC < {cutoff}")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    _legend_dots(ax, [("Multipass membrane", MP, "s", MP), ("Soluble", SOL, "s", "white")])
    return fig, d


# ------------------------------------------------------------------------------------------- agreement
def corr_panels(corrs: dict, predictors: list[str]):
    """Spearman rho between predictors over all substitutions: multipass, soluble, and the difference."""
    preds = [p for p in predictors if p in corrs[MEMBRANE].index]
    a, b = corrs[MEMBRANE].loc[preds, preds], corrs["SOLUBLE"].loc[preds, preds]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.0))
    mask = np.triu(np.ones((len(preds), len(preds)), bool))
    for ax, m, title, cmap, lim in ((axes[0], a, "Multipass membrane", SEQUENTIAL, (0, 1)),
                                    (axes[1], b, "Soluble", SEQUENTIAL, (0, 1)),
                                    (axes[2], a - b, "Difference (multipass − soluble)", DIVERGING, (-0.3, 0.3))):
        v = np.where(mask, np.nan, m.to_numpy(float))
        ax.imshow(v, cmap=cmap, vmin=lim[0], vmax=lim[1])
        for i in range(len(preds)):
            for j in range(i):
                x = v[i, j]
                if np.isfinite(x):
                    dark = (abs(x) > 0.55) if cmap is SEQUENTIAL else (abs(x) > 0.2)
                    ax.text(j, i, f"{x:+.2f}" if cmap is DIVERGING else f"{x:.2f}", ha="center", va="center",
                            fontsize=10, color="white" if dark else INK)
        ax.set_xticks(range(len(preds) - 1), [name(p) for p in preds[:-1]], rotation=45, ha="right", fontsize=11)
        ax.set_yticks(range(1, len(preds)), [name(p) for p in preds[1:]], fontsize=11)
        ax.set_xlim(-0.5, len(preds) - 1.5)
        ax.set_ylim(len(preds) - 0.5, 0.5)
        ax.set_title(title, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    diff = (a - b).where(~mask.T)
    return fig, diff.stack().sort_values(key=np.abs, ascending=False)


def calibration_bars(calib: pd.DataFrame, predictors: list[str]):
    w = calib[calib.predictor.isin(predictors)].pivot_table(index="predictor", columns="class", values="ece_prevalence_adjusted")
    w = w.dropna(subset=[MEMBRANE, "SOLUBLE"])
    w["ratio"] = w[MEMBRANE] / w["SOLUBLE"]
    w = w.sort_values("ratio")
    fig, ax = plt.subplots(figsize=WIDE)
    y = np.arange(len(w))
    ax.barh(y + 0.2, w[MEMBRANE], height=0.38, color=MP)
    ax.barh(y - 0.2, w["SOLUBLE"], height=0.38, color="white", edgecolor=SOL, lw=1.6)
    for i, (p, r) in enumerate(w.iterrows()):
        ax.text(max(r[MEMBRANE], r["SOLUBLE"]) + 0.002, i, f"{r.ratio:.1f}×", va="center", fontsize=12,
                color=MP if r.ratio > 1 else SUB)
    ax.set_yticks(y, [name(p) for p in w.index])
    ax.set_xlabel("Calibration error (ECE, prevalence-adjusted; map fitted on soluble proteins)")
    _legend_dots(ax, [("Multipass membrane", MP, "s", MP), ("Soluble", SOL, "s", "white")])
    return fig, w


def vus_bars(agree: pd.DataFrame, vep_label: str):
    d = agree.set_index("class").reindex([MEMBRANE, "SOLUBLE"]).dropna(how="all")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    cols = [("all pathogenic", MP), ("split", "#d8a25e"), ("all benign", SOL)]
    for i, (c, r) in enumerate(d.iterrows()):
        left = 0
        for k, col in cols:
            ax.barh(i, r[k], left=left, color=col, height=0.6)
            if r[k] > 0.06:
                ax.text(left + r[k] / 2, i, f"{r[k]:.0%}", ha="center", va="center", color="white", fontsize=13, fontweight="bold")
            left += r[k]
    ax.set_yticks(range(len(d)), [name(c) for c in d.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(f"VUS called by {vep_label} (soluble-tuned thresholds)")
    for k, col in cols:
        ax.bar(0, 0, color=col, label=k)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    return fig


# ------------------------------------------------------------------------------------------- structure
STRUCT_SHOW = {"h_depth_slope_kd": "Hydrophobicity rises toward core", "inversion_index_kd": "Hydrophobic surface (inversion)",
               "belt_contrast_scan_kd": "Hydrophobic belt", "frac_helix": "Helix fraction", "frac_strand": "Strand fraction",
               "cavity_frac": "Internal cavities", "contact_density_ca": "Packing (contact density)",
               "plddt_mean": "AlphaFold confidence", "a_over_c": "Elongation (a/c)", "sphericity_hull": "Sphericity"}


def structure_dots(contrasts: dict[str, pd.DataFrame], metrics: dict = STRUCT_SHOW, comparator: str = "SOLUBLE"):
    """Size-matched Cliff's delta, multipass vs soluble, one marker per organism."""
    ms = list(metrics)
    first = next(iter(contrasts.values()))
    base = first[first["class"] == comparator].set_index("metric").reindex(ms).cliffs_delta_size_matched
    ms = base.sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=TALL)
    styles = [(HUMAN_C, "o"), (YEAST_C, "D")]
    for (org, df), (col, mk) in zip(contrasts.items(), styles):
        v = df[df["class"] == comparator].set_index("metric").reindex(ms).cliffs_delta_size_matched
        ax.scatter(v, range(len(ms)), s=110, color=col, marker=mk, zorder=3, label=org)
    for i, m in enumerate(ms):
        vals = [df[df["class"] == comparator].set_index("metric").cliffs_delta_size_matched.get(m, np.nan) for df in contrasts.values()]
        ax.plot([0, np.nanmean(vals)], [i, i], color=LIGHT, lw=5, zorder=1)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(ms)), [metrics[m] for m in ms])
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Size-matched Cliff's δ, multipass vs soluble   (−1 always lower · +1 always higher)")
    ax.legend(loc="lower right")
    return fig


# ------------------------------------------------------------------------------------------- ThermoMPNN outliers
def thermo_scatter(t: pd.DataFrame, flag: pd.Series, other: str = "auc_esm1v", label_n: int = 8):
    d = t.dropna(subset=["auc_thermompnn", other])
    f = flag.reindex(d.index).fillna(False).astype(bool)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.scatter(d.loc[~f, "auc_thermompnn"], d.loc[~f, other], s=30, color=LIGHT, edgecolor="none", label="other multipass")
    ax.scatter(d.loc[f, "auc_thermompnn"], d.loc[f, other], s=46, color=MP, edgecolor="white", lw=0.5, label="bottom by ThermoMPNN AUC")
    ax.plot([0, 1], [0, 1], color=SUB, lw=1, ls=(0, (4, 3)))
    worst = d[f].nsmallest(label_n, "auc_thermompnn")
    for r in worst.itertuples():
        ax.annotate(r.gene, (r.auc_thermompnn, getattr(r, other)), xytext=(6, 2), textcoords="offset points", fontsize=11, color=INK)
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.2, 1.02)
    ax.set_xlabel("ThermoMPNN AUC (per protein)")
    ax.set_ylabel(f"{name(other.replace('auc_', ''))} AUC (per protein)")
    ax.legend(loc="lower right")
    return fig


def profile(t: pd.DataFrame, flag: pd.Series, cols: list[str], xlabel: str, ref: float | None = 0.0):
    """Median and IQR of each column for bottom proteins vs the rest."""
    cols = [c for c in cols if c in t and t[c].notna().sum() > 5]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(cols) + 1.8))
    for i, c in enumerate(cols):
        for k, (sel, col, fill) in enumerate(((flag, MP, MP), (~flag, SOL, "white"))):
            v = t.loc[sel, c].dropna()
            if not len(v):
                continue
            y = i + (0.17 if k == 0 else -0.17)
            ax.plot([v.quantile(0.25), v.quantile(0.75)], [y, y], color=col, lw=2)
            ax.scatter(v.median(), y, s=80, color=fill, edgecolor=col, lw=1.8, zorder=3)
    if ref is not None:
        ax.axvline(ref, color=LIGHT, lw=1, zorder=0)
    ax.set_yticks(range(len(cols)), [name(c[4:]) if c[:4] in ("rho_", "auc_") else flabel(c) for c in cols])
    ax.set_xlabel(xlabel)
    _legend_dots(ax, [("Bottom proteins", MP, "o", MP), ("Other multipass", SOL, "o", "white")], loc="lower right")
    return fig


def effects(cmp: pd.DataFrame, labels: dict, top: int = 10):
    d = cmp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(d) + 1.8))
    col = [MP if v > 0 else SOL for v in d.cliffs_delta]
    ax.barh(range(len(d)), d.cliffs_delta, color=col, height=0.6)
    for i, r in enumerate(d.itertuples()):
        q = "q < 0.001" if r.q < 0.001 else f"q = {r.q:.2g}"
        ax.text(1.08, i, f"{fnum(r.median_bottom)} vs {fnum(r.median_rest)}", va="center", ha="left", fontsize=12, color=INK,
                transform=ax.get_yaxis_transform())
        ax.text(1.36, i, q, va="center", ha="left", fontsize=11, color=SUB, transform=ax.get_yaxis_transform())
    ax.text(1.08, len(d) - 0.3, "median: bottom vs rest", ha="left", fontsize=11, color=SUB, transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(d)), [flabel(f, labels) for f in d.feature])
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Cliff's δ, bottom proteins vs other multipass")
    return fig


def family_bars(fc: pd.DataFrame, top: int = 8):
    d = fc.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(d) + 1.6))
    y = np.arange(len(d))
    ax.barh(y + 0.2, d.frac_bottom, height=0.38, color=MP, label="Bottom proteins")
    ax.barh(y - 0.2, d.frac_all, height=0.38, color="white", edgecolor=SUB, lw=1.4, label="All multipass")
    for i, r in enumerate(d.itertuples()):
        ax.text(max(r.frac_bottom, r.frac_all) + 0.005, i, f"{int(r.bottom)} / {int(r.all)}", va="center", fontsize=11, color=SUB)
    ax.set_yticks(y, d.index)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of proteins")
    ax.legend(loc="lower right")
    return fig


def agreement_hist(t_all: pd.DataFrame):
    fig, ax = plt.subplots(figsize=WIDE)
    bins = np.linspace(-0.4, 1.0, 57)
    for c, col, fill in ((MEMBRANE, MP, True), ("SOLUBLE", SOL, False)):
        v = t_all[t_all.class_annotation == c].agreement.dropna()
        ax.hist(v, bins=bins, density=True, histtype="stepfilled" if fill else "step", color=col, alpha=0.85 if fill else 1,
                lw=2, label=f"{name(c)} (median {v.median():.2f})")
    ax.set_xlabel("ThermoMPNN agreement with sequence predictors (median per-protein Spearman ρ)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    return fig
