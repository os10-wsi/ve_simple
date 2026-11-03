"""AlphaMissense scores from the AlphaFold DB per-protein substitution table."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_VAR_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def read_alphamissense(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    parsed = df["protein_variant"].str.extract(_VAR_RE)
    out = pd.DataFrame({
        "position": parsed[1].astype(int),
        "wt": parsed[0],
        "mut": parsed[2],
        "alphamissense": df["am_pathogenicity"].astype(float),
        "alphamissense_class": df["am_class"],
    })
    return out.dropna(subset=["position"])
