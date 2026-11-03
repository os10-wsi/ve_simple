"""Amino-acid property tables and substitution matrix used by the physico-chemical predictors."""
from __future__ import annotations

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
HUMAN_TAXID = 9606

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # common non-standard residues mapped to their parent amino acid
    "MSE": "M", "SEC": "C", "PYL": "K", "HSD": "H", "HSE": "H", "HSP": "H", "HID": "H", "HIE": "H",
}

# Residue side-chain volumes in A^3 (Zamyatnin, 1972).
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4,
    "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9,
    "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

# Kyte-Doolittle hydropathy index.
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Nominal side-chain charge at physiological pH (His treated as neutral).
CHARGE = {aa: 0.0 for aa in AMINO_ACIDS}
CHARGE.update({"K": 1.0, "R": 1.0, "D": -1.0, "E": -1.0, "H": 0.0})

# Maximum accessible surface area (A^2), theoretical values from Tien et al. 2013.
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0, "Q": 225.0, "E": 223.0,
    "G": 104.0, "H": 224.0, "I": 197.0, "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0,
    "P": 159.0, "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}

_BLOSUM62_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_ROWS = [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
]
BLOSUM62 = {
    (a, b): _BLOSUM62_ROWS[i][j]
    for i, a in enumerate(_BLOSUM62_ORDER)
    for j, b in enumerate(_BLOSUM62_ORDER)
}

# Every predictor column that can be evaluated by ROC-AUC, with the sign that orients it so
# that a HIGHER oriented score means MORE likely pathogenic.  AUCs in the output tables are
# computed on the oriented score, so values > 0.5 always mean "better than random".
PREDICTOR_DIRECTIONS = {
    "alphamissense": +1,          # AlphaMissense pathogenicity probability
    "saprot": -1,                 # log-odds mut/wt; negative = damaging
    "esm1v": -1,                  # log-odds mut/wt; negative = damaging
    "foldx": +1,                  # ddG (kcal/mol); positive = destabilising
    "thermompnn": +1,             # ddG (kcal/mol); positive = destabilising
    "blosum62": -1,               # low substitution score = non-conservative
    "delta_volume": +1,           # signed (mut - wt)
    "delta_charge": +1,           # signed (mut - wt)
    "delta_hydrophobicity": +1,   # signed (mut - wt)
    "abs_delta_volume": +1,
    "abs_delta_charge": +1,
    "abs_delta_hydrophobicity": +1,
    "solvent_exposure": -1,       # relative SASA; buried (low) = more often pathogenic
}
SCORE_COLUMNS = list(PREDICTOR_DIRECTIONS)
