"""Protein-level membrane/soluble classes and domain labels."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MEMBRANE_CLASSES = ("MULTIPASS", "SINGLE_PASS", "BETA_BARREL_MEMBRANE")
# Mitochondrial outer-membrane beta barrels named in the spec (UniProt SIMILARITY family names).
_OMM_BARREL = re.compile(r"mitochondrial porin|tom40|sam50|omp85", re.I)

# Structural belt detector (CHAIN_TRIMMED unit, Kyte-Doolittle, surface shell): the most hydrophobic
# 30 A slab over 200 directions (voxel.slab_scan) must be at least BELT_MIN_CONTRAST more hydrophobic
# than the rest of the surface. The slab thickness encodes the 25-35 A bilayer core, so no separate width
# condition is applied (the run-above-median width did not separate classes on Stage 0).
# Threshold chosen on the Stage 0 sample (class medians 1.1 soluble vs 2.5 multipass): see docs/stage0.md.
BELT_MIN_CONTRAST = 2.0


def class_annotation(row) -> tuple[str, str]:
    """Class from UniProt annotation alone, with the reason."""
    n_tm = row.n_tm_helical + row.n_tm_beta
    if row.n_tm_beta > 0:
        return "BETA_BARREL_MEMBRANE", "beta-stranded TM features"
    if _OMM_BARREL.search(row.similarity or ""):
        return "BETA_BARREL_MEMBRANE", "mitochondrial outer-membrane barrel family"
    if row.reviewed and row.n_tm_helical >= 2:
        return "MULTIPASS", f"{row.n_tm_helical} helical TM features"
    if n_tm == 1:
        return "SINGLE_PASS", "1 TM feature"
    if n_tm == 0 and row.n_intramem == 0 and row.n_lipid == 0 and not row.location_mentions_membrane:
        return ("SOLUBLE", "no TM/intramembrane/lipid; location without 'membrane'") if row.reviewed \
            else ("AMBIGUOUS", "unreviewed")
    reasons = []
    if n_tm:
        reasons.append("TM features but unreviewed")
    if row.n_intramem:
        reasons.append("intramembrane feature")
    if row.n_lipid:
        reasons.append("lipid anchor")
    if row.location_mentions_membrane:
        reasons.append("location mentions membrane")
    return "AMBIGUOUS", "; ".join(reasons)


def classify(ann: pd.DataFrame) -> pd.DataFrame:
    res = [class_annotation(r) for r in ann.itertuples()]
    out = ann[["accession", "taxid", "gene", "protein_name", "length", "reviewed"]].copy()
    out["class_annotation"] = [c for c, _ in res]
    out["class_reason"] = [r for _, r in res]
    return out


def belt_detected(width: float, contrast: float) -> bool:
    """`width` is accepted for reporting symmetry but not used (see BELT_MIN_CONTRAST)."""
    return bool(np.isfinite(contrast) and contrast >= BELT_MIN_CONTRAST)


def add_final_class(classes: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """class_final: annotation class, moved to AMBIGUOUS where the structural belt detector disagrees.
    Applied to MULTIPASS, BETA_BARREL_MEMBRANE and SOLUBLE; SINGLE_PASS chains are dominated by their
    soluble part, so they are left unchanged."""
    t = metrics[(metrics["unit"] == "CHAIN_TRIMMED") & (metrics["mask_variant"] == "default")]
    t = t.set_index("accession")
    belt = {a: belt_detected(r.get("belt_width_scan_A_kd", np.nan), r.get("belt_contrast_scan_kd", np.nan))
            for a, r in t.iterrows()}
    out = classes.copy()
    out["belt_detected"] = out["accession"].map(belt)
    final = []
    for c, b in zip(out["class_annotation"], out["belt_detected"]):
        if b is None or (isinstance(b, float) and np.isnan(b)):
            final.append(c)                       # no trimmed unit -> cannot cross-check
        elif c in ("MULTIPASS", "BETA_BARREL_MEMBRANE") and not b:
            final.append("AMBIGUOUS")
        elif c == "SOLUBLE" and b:
            final.append("AMBIGUOUS")
        else:
            final.append(c)
    out["class_final"] = final
    return out


def domain_label(protein_class: str, n_tm_overlap: int, frac_tm: float) -> str:
    if n_tm_overlap >= 2 or frac_tm >= 0.40:
        return "MEMBRANE_DOMAIN"
    if protein_class in ("MULTIPASS", "SINGLE_PASS") and frac_tm < 0.10:
        return "SOLUBLE_DOMAIN_OF_MEMBRANE_PROTEIN"
    if protein_class == "SOLUBLE":
        return "SOLUBLE_DOMAIN"
    return "MIXED"
