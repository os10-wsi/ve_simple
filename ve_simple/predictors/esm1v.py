"""ESM-1v masked-marginal variant scores (ensemble of up to five models)."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..constants import AMINO_ACIDS
from .lm_common import logodds_to_frame, masked_marginals, resolve_device

log = logging.getLogger(__name__)

ESM1V_MODELS = [f"facebook/esm1v_t33_650M_UR90S_{i}" for i in range(1, 6)]


class ESM1vScorer:
    """Loads one ESM-1v model at a time; call `score(sequence)` for a log-odds matrix."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: Optional[int] = None):
        import torch
        from transformers import AutoTokenizer, EsmForMaskedLM

        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size or (32 if self.device == "cuda" else 4)
        log.info("Loading %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmForMaskedLM.from_pretrained(model_name)
        if self.device == "cuda":
            self.model = self.model.half()
        self.model.to(self.device).eval()
        torch.set_grad_enabled(False)

    def score_matrix(self, sequence: str, positions=None) -> np.ndarray:
        seq = "".join(aa if aa in AMINO_ACIDS else "X" for aa in sequence)
        return masked_marginals(
            list(seq), tokenizer=self.tokenizer, model=self.model, device=self.device,
            token_for=lambda i, aa: aa, mask_token_for=lambda i: self.tokenizer.mask_token,
            positions=positions, batch_size=self.batch_size,
        )

    def close(self):
        import gc, torch
        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


def esm1v_frame(matrices: list[np.ndarray], sequence: str) -> pd.DataFrame:
    """Average the per-model log-odds matrices and return a long table with column `esm1v`."""
    stack = np.stack(matrices)
    mean = np.nanmean(stack, axis=0) if len(matrices) > 1 else stack[0]
    df = logodds_to_frame(mean, sequence, "esm1v")
    df["esm1v_n_models"] = len(matrices)
    return df
