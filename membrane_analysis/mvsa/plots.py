"""Figure helpers with one visual language: classes always in the same colour, MULTIPASS highlighted."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .groups import CLASS_COLOR, CLASS_LABEL, PREDICTOR_LABEL

FIG_DIR = Path("figures")
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def setup(fig_dir: str | Path = "figures") -> None:
    global FIG_DIR
    FIG_DIR = Path(fig_dir)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
    })


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")


def label(x: str) -> str:
    return PREDICTOR_LABEL.get(x, CLASS_LABEL.get(x, x))


def forest(df: pd.DataFrame, row: str, group: str, value: str = "auc", lo: str = "ci_low", hi: str = "ci_high",
           ref: float | None = 0.5, ax=None, groups: list | None = None, xlabel: str = "AUC", title: str = ""):
    """Point estimates with 95% CIs: one row per `row` value, one dodged mark per `group` value."""
    groups = groups or list(dict.fromkeys(df[group]))
    rows = list(dict.fromkeys(df[row]))
    ax = ax or plt.subplots(figsize=(6, 0.42 * len(rows) + 1))[1]
    step = 0.7 / max(len(groups), 1)
    for j, g in enumerate(groups):
        sub = df[df[group] == g].set_index(row)
        y = np.arange(len(rows)) + (j - (len(groups) - 1) / 2) * step
        v = sub.reindex(rows)
        ax.errorbar(v[value], y, xerr=[v[value] - v[lo], v[hi] - v[value]], fmt="o", ms=5,
                    color=CLASS_COLOR.get(g, INK2), ecolor=CLASS_COLOR.get(g, INK2), elinewidth=1.4, capsize=0,
                    label=CLASS_LABEL.get(g, g), mec="white", mew=0.8)
    if ref is not None:
        ax.axvline(ref, color=AXIS, lw=1, zorder=0)
    ax.set_yticks(range(len(rows)), [label(r) for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=min(3, len(groups)), fontsize=8)
    return ax


def class_violin(df: pd.DataFrame, value: str, classes: list, ax=None, ylabel: str = "", title: str = "",
                 group: str = "protein_class", ref: float | None = None):
    ax = ax or plt.subplots(figsize=(5, 3.2))[1]
    data = [df.loc[df[group] == c, value].dropna().to_numpy() for c in classes]
    keep = [i for i, d in enumerate(data) if len(d) >= 3]
    if keep:
        parts = ax.violinplot([data[i] for i in keep], positions=keep, showmedians=True, widths=0.8)
        for b, i in zip(parts["bodies"], keep):
            b.set_facecolor(CLASS_COLOR.get(classes[i], INK2))
            b.set_edgecolor("white")
            b.set_alpha(0.75)
        for k in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[k].set_color(INK2)
            parts[k].set_linewidth(0.8)
    ax.set_xticks(range(len(classes)), [f"{CLASS_LABEL.get(c, c)}\n(n={len(d)})" for c, d in zip(classes, data)],
                  fontsize=7)
    if ref is not None:
        ax.axhline(ref, color=AXIS, lw=1, zorder=0)
    ax.set_ylabel(ylabel or label(value))
    ax.set_title(title, loc="left")
    return ax


def ecdf(df: pd.DataFrame, value: str, classes: list, ax=None, group: str = "protein_class", xlabel: str = ""):
    ax = ax or plt.subplots(figsize=(4.5, 3.2))[1]
    for c in classes:
        v = np.sort(df.loc[df[group] == c, value].dropna().to_numpy())
        if len(v):
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=CLASS_COLOR.get(c, INK2), lw=2,
                    label=f"{CLASS_LABEL.get(c, c)} (n={len(v)})")
    ax.set_xlabel(xlabel or label(value))
    ax.set_ylabel("Cumulative fraction")
    ax.legend(fontsize=7)
    return ax


def heatmap(m: pd.DataFrame, ax=None, title: str = "", vmin=-1, vmax=1, cmap="RdBu_r", fmt="{:.2f}"):
    ax = ax or plt.subplots(figsize=(0.55 * len(m.columns) + 1.5, 0.45 * len(m) + 1))[1]
    im = ax.imshow(m.to_numpy(float), vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(m.columns)), [label(c) for c in m.columns], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(m.index)), [label(c) for c in m.index], fontsize=7)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.6 * max(abs(vmin), abs(vmax)) else INK)
    ax.set_title(title, loc="left")
    plt.colorbar(im, ax=ax, fraction=0.04)
    return ax
