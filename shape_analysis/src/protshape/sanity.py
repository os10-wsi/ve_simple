"""Stage-4 sanity set: proteins with known shapes, and the checks they must pass."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .classify import belt_detected

# (label, accession, unit, how to pick the domain, expectation)
SANITY = [
    ("lysozyme-like globular domain", "P61626", "DOMAIN", None, "a/c 1.2-1.5, Rg/Rmin ~1, no belt"),
    ("spectrin repeat (rod)", "Q13813", "DOMAIN", "1.20.58.60", "rod: a/c > 3, kappa2 high"),
    ("7TM GPCR (ADRB2)", "P07550", "CHAIN_TRIMMED", None, "belt detected"),
    ("12TM transporter (GLUT1)", "P11166", "CHAIN_TRIMMED", None, "belt detected"),
    ("mitochondrial VDAC barrel", "P21796", "CHAIN_TRIMMED", None, "beta-barrel class (belt reported, not required)"),
    ("long coiled coil (TPM1)", "P09493", "CHAIN_TRIMMED:no_ted", None, "rod: a/c > 5, helix > 90% (TED has no domain)"),
    ("intrinsically disordered (SNCA)", "P37840", "CHAIN_FULL", None, "Rg/Rmin >> 1 on CHAIN_FULL; little survives trimming"),
]
SANITY_IDS = [a for _, a, *_ in SANITY]
COLS = ["n_res", "rg_ca", "rg_ca_over_rmin_empirical", "a_over_c", "kappa2", "prolateness", "solidity_ses",
        "sphericity_hull", "frac_helix", "inversion_index_kd", "belt_width_scan_A_kd", "belt_contrast_scan_kd",
        "belt_normal_angle_pa1_kd"]


def _pick(m: pd.DataFrame, acc: str, unit: str, cath: str | None) -> pd.Series | None:
    unit, _, variant = unit.partition(":")
    sub = m[(m.accession == acc) & (m.unit == unit) & (m.mask_variant == (variant or "default"))]
    if unit == "DOMAIN" and cath:
        sub = sub[sub.cath_label.astype(str).str.startswith(cath)]
    if sub.empty:
        return None
    return sub.sort_values("n_res", ascending=False).iloc[0]


def sanity_table(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, acc, unit, cath, expect in SANITY:
        r = _pick(m, acc, unit, cath)
        if r is None:
            rows.append({"protein": label, "accession": acc, "unit": unit, "expectation": expect, "check": "MISSING"})
            continue
        belt = belt_detected(r.get("belt_width_scan_A_kd", np.nan), r.get("belt_contrast_scan_kd", np.nan))
        ok = {
            "P61626": 1.2 <= r.a_over_c <= 1.5 and 0.9 <= r.rg_ca_over_rmin_empirical <= 1.1 and not belt,
            "Q13813": r.a_over_c > 3 and r.kappa2 > 0.5,
            "P07550": belt, "P11166": belt,
            "P21796": r.protein_class == "BETA_BARREL_MEMBRANE",
            "P09493": r.a_over_c > 5 and r.frac_helix > 0.9,
            "P37840": r.rg_ca_over_rmin_empirical > 1.5,
        }[acc]
        row = {"protein": label, "accession": acc, "unit": unit, "domain": r.get("domain_id", ""),
               "expectation": expect, "check": "PASS" if ok else "FAIL", "belt": belt}
        row.update({c: round(float(r.get(c, np.nan)), 3) for c in COLS})
        if acc == "P37840":
            tr = m[(m.accession == acc) & (m.unit == "CHAIN_TRIMMED") & (m.mask_variant == "default")]
            row["note"] = f"CHAIN_TRIMMED keeps {int(tr.n_res.iloc[0]) if len(tr) else 0} of {int(r.n_res)} residues"
        rows.append(row)
    # disorder inflation of Rg/Rmin: CHAIN_FULL vs CHAIN_TRIMMED for every protein in the sanity set
    return pd.DataFrame(rows)


def disorder_inflation(m: pd.DataFrame) -> pd.DataFrame:
    f = m[m.unit == "CHAIN_FULL"].set_index("accession")["rg_ca_over_rmin_empirical"]
    t = m[(m.unit == "CHAIN_TRIMMED") & (m.mask_variant == "default")].set_index("accession")["rg_ca_over_rmin_empirical"]
    d = pd.DataFrame({"full": f, "trimmed": t}).dropna()
    d["ratio_full_over_trimmed"] = d["full"] / d["trimmed"]
    return d
