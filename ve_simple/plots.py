"""Multi-panel PDF of per-protein ROC curves (one panel per protein, one curve per predictor)."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import pandas as pd  # noqa: E402

from .evaluate import roc_points  # noqa: E402

log = logging.getLogger(__name__)

# Fixed identity -> (colour, line style). Colours are the validated 8-hue categorical palette
# (colour-vision-deficiency checked); the dash pattern is the secondary encoding for the
# physico-chemical deltas so that identity never relies on colour alone.
PREDICTOR_STYLE = {
    "alphamissense":            ("#2a78d6", "-"),
    "esm1v":                    ("#eb6834", "-"),
    "saprot":                   ("#1baf7a", "-"),
    "thermompnn":               ("#eda100", "-"),
    "foldx":                    ("#e87ba4", "-"),
    "solvent_exposure":         ("#008300", "-"),
    "blosum62":                 ("#6250d6", "-"),
    "abs_delta_hydrophobicity": ("#e34948", "-"),
    "abs_delta_volume":         ("#2a78d6", "--"),
    "abs_delta_charge":         ("#eb6834", "--"),
    "delta_hydrophobicity":     ("#e34948", ":"),
    "delta_volume":             ("#1baf7a", ":"),
    "delta_charge":             ("#eda100", ":"),
}
DEFAULT_PLOT_PREDICTORS = ["alphamissense", "esm1v", "saprot", "thermompnn", "foldx", "solvent_exposure",
                           "blosum62", "abs_delta_hydrophobicity", "abs_delta_volume", "abs_delta_charge"]
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"


def _panel(ax, variants: pd.DataFrame, title: str, predictors: list[str]) -> int:
    n_curves = 0
    for pred in predictors:
        if pred not in variants.columns:
            continue
        pts = roc_points(variants, pred)
        if pts is None:
            continue
        fpr, tpr, auc = pts
        colour, ls = PREDICTOR_STYLE.get(pred, ("#52514e", "-"))
        ax.plot(fpr, tpr, color=colour, linestyle=ls, linewidth=1.6, label=f"{pred} ({auc:.2f})")
        n_curves += 1
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8, color=TEXT, loc="left")
    ax.set_xlabel("False positive rate", fontsize=7, color=TEXT2)
    ax.set_ylabel("True positive rate", fontsize=7, color=TEXT2)
    ax.tick_params(labelsize=6, colors=TEXT2, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    if n_curves:
        ax.legend(fontsize=5, frameon=False, loc="lower right", handlelength=2.2, labelcolor=TEXT2)
    else:
        ax.text(0.5, 0.5, "no pathogenic/benign pairs", ha="center", va="center", fontsize=7, color=TEXT2)
    return n_curves


def roc_pdf(variants: pd.DataFrame, proteins: pd.DataFrame, out_pdf: Path,
            predictors: Optional[list[str]] = None, ncols: int = 3, nrows: int = 3) -> Path:
    """Write one PDF with a grid of ROC panels (one per human protein with ClinVar labels)."""
    predictors = predictors or DEFAULT_PLOT_PREDICTORS
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    meta = proteins.set_index("accession")
    accs = [a for a in meta.index if a in set(variants["accession"])]
    per_page = ncols * nrows
    n_pages = max(1, math.ceil(len(accs) / per_page))
    with PdfPages(out_pdf) as pdf:
        for page in range(n_pages):
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.4))
            axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
            chunk = accs[page * per_page:(page + 1) * per_page]
            for ax, acc in zip(axes, chunk):
                sub = variants[variants["accession"] == acc]
                m = meta.loc[acc]
                n_p = int((sub["clinvar_label"] == "pathogenic").sum())
                n_b = int((sub["clinvar_label"] == "benign").sum())
                title = f"{acc} {m.get('gene', '') or ''}  (P={n_p}, B={n_b})"
                _panel(ax, sub, title, predictors)
            for ax in axes[len(chunk):]:
                ax.axis("off")
            fig.suptitle("ROC curves: ClinVar pathogenic vs benign (AUC in legend, scores oriented so higher = pathogenic)",
                         fontsize=8, color=TEXT2, x=0.01, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)
    log.info("Wrote %s (%d proteins, %d page(s))", out_pdf, len(accs), n_pages)
    return out_pdf
