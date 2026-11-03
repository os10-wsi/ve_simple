"""SaProt structure-aware masked-marginal scores (AlphaFold model + 3Di tokens).

The 3Di structural alphabet is computed with the pure-Python `mini3di` port of Foldseek, or
with the `foldseek` binary if it is on PATH (set VE_SIMPLE_FOLDSEEK=/path/to/foldseek to force).
Following the SaProt authors, structure tokens at residues with pLDDT < `plddt_mask` are masked
('#') so that uncertain structure does not mislead the model.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..alphafold import read_pdb_chain
from ..constants import AMINO_ACIDS
from .lm_common import logodds_to_frame, masked_marginals, resolve_device

log = logging.getLogger(__name__)

SAPROT_MODEL = "westlake-repl/SaProt_650M_AF2"


def foldseek_binary() -> Optional[str]:
    cand = os.environ.get("VE_SIMPLE_FOLDSEEK") or shutil.which("foldseek")
    return cand if cand and os.path.exists(cand) else None


def threedi_sequence(pdb_path: Path) -> str:
    """Return the lowercase 3Di string for the first chain of `pdb_path`."""
    exe = foldseek_binary()
    if exe:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.tsv"
            subprocess.run([exe, "structureto3didescriptor", str(pdb_path), str(out)],
                           check=True, capture_output=True)
            line = out.read_text().strip().split("\n")[0].split("\t")
            # columns: name, aa sequence, 3Di sequence, descriptor...
            return line[2].lower()
    import mini3di

    chain, seq, _, _ = read_pdb_chain(pdb_path)
    encoder = mini3di.Encoder()
    states = encoder.encode_chain(chain)
    return encoder.build_sequence(states).lower()


def structure_aware_tokens(sequence: str, threedi: str, plddt: np.ndarray, plddt_mask: float = 70.0) -> list[str]:
    if len(sequence) != len(threedi):
        raise ValueError(f"sequence length {len(sequence)} != 3Di length {len(threedi)}")
    toks = []
    for aa, s, pl in zip(sequence, threedi, plddt):
        s = "#" if (plddt_mask and pl < plddt_mask) else s
        toks.append((aa if aa in AMINO_ACIDS else "X") + s)
    return toks


class SaProtScorer:
    def __init__(self, model_name: str = SAPROT_MODEL, device: str = "auto",
                 batch_size: Optional[int] = None, plddt_mask: float = 70.0):
        import torch
        from transformers import EsmForMaskedLM, EsmTokenizer

        self.device = resolve_device(device)
        self.batch_size = batch_size or (32 if self.device == "cuda" else 4)
        self.plddt_mask = plddt_mask
        log.info("Loading %s on %s", model_name, self.device)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmForMaskedLM.from_pretrained(model_name)
        if self.device == "cuda":
            self.model = self.model.half()
        self.model.to(self.device).eval()
        torch.set_grad_enabled(False)

    def score_structure(self, pdb_path: Path, positions=None) -> tuple[np.ndarray, str, str]:
        """Return (log-odds matrix, model sequence, 3Di string) for an AlphaFold PDB."""
        _, seq, _, plddt = read_pdb_chain(pdb_path)
        threedi = threedi_sequence(pdb_path)
        tokens = structure_aware_tokens(seq, threedi, plddt, self.plddt_mask)
        mat = masked_marginals(
            tokens, tokenizer=self.tokenizer, model=self.model, device=self.device,
            token_for=lambda i, aa: aa + tokens[i][1],
            mask_token_for=lambda i: "#" + tokens[i][1],
            positions=positions, batch_size=self.batch_size,
        )
        return mat, seq, threedi

    def score_frame(self, pdb_path: Path) -> pd.DataFrame:
        mat, seq, _ = self.score_structure(pdb_path)
        return logodds_to_frame(mat, seq, "saprot")

    def close(self):
        import gc, torch
        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
