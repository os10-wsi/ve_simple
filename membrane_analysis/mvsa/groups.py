"""Protein classes, predictor groups, orientation and colours shared by every analysis."""
from __future__ import annotations

# Classes from shape_analysis classify (UniProt annotation). MULTIPASS is the membrane group.
MEMBRANE = "MULTIPASS"
CLASS_ORDER = ["MULTIPASS", "SINGLE_PASS", "BETA_BARREL_MEMBRANE", "SOLUBLE", "AMBIGUOUS"]
CLASS_LABEL = {"MULTIPASS": "Multipass membrane", "SINGLE_PASS": "Single-pass membrane",
               "BETA_BARREL_MEMBRANE": "Beta-barrel membrane", "SOLUBLE": "Soluble", "AMBIGUOUS": "Ambiguous",
               "NON_MULTIPASS": "All non-multipass"}
# Validated categorical palette (dataviz reference); MULTIPASS gets the strongest, warm hue.
CLASS_COLOR = {"MULTIPASS": "#eb6834", "SINGLE_PASS": "#eda100", "BETA_BARREL_MEMBRANE": "#6250d6",
               "SOLUBLE": "#2a78d6", "AMBIGUOUS": "#8f8d85", "NON_MULTIPASS": "#1baf7a"}

# Predictors (columns of ve_simple variant_scores) with their orientation: multiply by the sign so that
# HIGHER always means MORE likely pathogenic. Mirrors ve_simple.constants.PREDICTOR_DIRECTIONS.
DIRECTION = {
    "alphamissense": +1, "esm1v": -1, "saprot": -1, "thermompnn": +1, "foldx": +1, "blosum62": -1,
    "abs_delta_hydrophobicity": +1, "abs_delta_volume": +1, "abs_delta_charge": +1,
    "delta_hydrophobicity": +1, "delta_volume": +1, "delta_charge": +1, "solvent_exposure": -1,
}
PREDICTOR_GROUP = {
    "alphamissense": "Variant-effect predictor", "esm1v": "Variant-effect predictor", "saprot": "Variant-effect predictor",
    "thermompnn": "Stability predictor", "foldx": "Stability predictor",
    "blosum62": "Substitution / physico-chemical", "abs_delta_hydrophobicity": "Substitution / physico-chemical",
    "abs_delta_volume": "Substitution / physico-chemical", "abs_delta_charge": "Substitution / physico-chemical",
    "delta_hydrophobicity": "Substitution / physico-chemical", "delta_volume": "Substitution / physico-chemical",
    "delta_charge": "Substitution / physico-chemical", "solvent_exposure": "Structural context",
}
PREDICTOR_LABEL = {
    "alphamissense": "AlphaMissense", "esm1v": "ESM-1v", "saprot": "SaProt", "thermompnn": "ThermoMPNN ddG",
    "foldx": "FoldX ddG", "blosum62": "BLOSUM62", "abs_delta_hydrophobicity": "|delta hydrophobicity|",
    "abs_delta_volume": "|delta volume|", "abs_delta_charge": "|delta charge|",
    "delta_hydrophobicity": "delta hydrophobicity", "delta_volume": "delta volume", "delta_charge": "delta charge",
    "solvent_exposure": "Burial (rel. SASA, inverted)",
}
PREDICTORS = list(DIRECTION)
# predictors shown in the headline figures (signed deltas are kept for the explorer)
MAIN_PREDICTORS = ["alphamissense", "esm1v", "saprot", "thermompnn", "foldx", "blosum62",
                   "abs_delta_hydrophobicity", "abs_delta_volume", "abs_delta_charge", "solvent_exposure"]

# Published AlphaMissense cut-offs (Cheng et al. 2023): likely benign < 0.34, likely pathogenic > 0.564.
PUBLISHED_THRESHOLDS = {"alphamissense": (0.34, 0.564)}

# Residue locations used for the "where in the protein" analyses
LOCATIONS_MEMBRANE = ["TM lipid-facing", "TM buried", "Juxtamembrane", "Extramembrane exposed", "Extramembrane buried"]
LOCATIONS_OTHER = ["Exposed", "Buried"]
EXPOSED_RSA = 0.20     # relative SASA at or above which a residue counts as exposed (lipid-facing inside a TM)
JUXTA = 5              # residues either side of a TM segment


def oriented(df, predictor):
    """Predictor scores oriented so that higher = more likely pathogenic."""
    return df[predictor] * DIRECTION[predictor]
