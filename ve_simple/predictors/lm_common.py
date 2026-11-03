"""Shared masked-marginal scoring for ESM-style masked language models (HF transformers)."""
from __future__ import annotations

import logging
import os
from typing import Callable, Iterable, Sequence

# Hugging Face's Xet-backed downloads need an extra CDN host (cas-server.xethub.hf.co) that is
# often blocked by institutional proxies; plain HTTPS downloads work everywhere.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np

from ..constants import AMINO_ACIDS

log = logging.getLogger(__name__)

MAX_RESIDUES = 1022  # ESM-1v / ESM-2 positional limit (1024 tokens incl. <cls> and <eos>)


def resolve_device(device: str = "auto") -> str:
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def window_bounds(length: int, pos0: int, max_len: int = MAX_RESIDUES) -> tuple[int, int]:
    """Half-open [start, end) window of at most `max_len` residues containing 0-based `pos0`."""
    if length <= max_len:
        return 0, length
    start = max(0, pos0 - max_len // 2)
    end = start + max_len
    if end > length:
        end = length
        start = end - max_len
    return start, end


def masked_marginals(
    tokens: Sequence[str],
    *,
    tokenizer,
    model,
    device: str,
    token_for: Callable[[int, str], str],
    mask_token_for: Callable[[int], str],
    positions: Iterable[int] | None = None,
    batch_size: int = 8,
    max_len: int = MAX_RESIDUES,
) -> np.ndarray:
    """Log-odds matrix [L, 20]: log p(mut) - log p(wt) at each masked position.

    `tokens[i]` is the input token for residue i (e.g. 'M' for ESM-1v, 'Md' for SaProt).
    `token_for(i, aa)` gives the vocabulary token of amino acid `aa` at residue i;
    `mask_token_for(i)` gives the mask token at residue i.  Positions whose wild-type token is
    not in the vocabulary are left as NaN.
    """
    import torch

    L = len(tokens)
    positions = list(range(L)) if positions is None else list(positions)
    out = np.full((L, len(AMINO_ACIDS)), np.nan, dtype=np.float32)
    aa_token_ids_cache: dict[tuple, list[int]] = {}
    unk = tokenizer.unk_token_id

    # group positions by window so that batches share the same context tokens
    groups: dict[tuple[int, int], list[int]] = {}
    for p in positions:
        groups.setdefault(window_bounds(L, p, max_len), []).append(p)

    model.eval()
    with torch.no_grad():
        for (start, end), plist in groups.items():
            ctx = list(tokens[start:end])
            base_ids = tokenizer.convert_tokens_to_ids(ctx)
            cls_id, eos_id = tokenizer.cls_token_id, tokenizer.eos_token_id
            base = [cls_id] + base_ids + [eos_id]
            for b in range(0, len(plist), batch_size):
                chunk = plist[b:b + batch_size]
                ids = torch.tensor([base] * len(chunk), dtype=torch.long)
                for r, p in enumerate(chunk):
                    ids[r, p - start + 1] = tokenizer.convert_tokens_to_ids(mask_token_for(p))
                ids = ids.to(device)
                logits = model(input_ids=ids).logits
                logp = torch.log_softmax(logits.float(), dim=-1)
                for r, p in enumerate(chunk):
                    row = logp[r, p - start + 1]
                    key = tuple(token_for(p, aa) for aa in AMINO_ACIDS)
                    if key not in aa_token_ids_cache:
                        aa_token_ids_cache[key] = tokenizer.convert_tokens_to_ids(list(key))
                    aa_ids = aa_token_ids_cache[key]
                    wt_tok = tokens[p]
                    wt_id = tokenizer.convert_tokens_to_ids(wt_tok)
                    if wt_id == unk or wt_id is None:
                        continue
                    vals = row[aa_ids].cpu().numpy() - row[wt_id].item()
                    vals = np.where(np.asarray(aa_ids) == unk, np.nan, vals)
                    out[p] = vals
    return out


def logodds_to_frame(matrix: np.ndarray, sequence: str, colname: str, positions_1based=None):
    """Turn an [L, 20] log-odds matrix into a long (position, wt, mut, score) table."""
    import pandas as pd

    rows = []
    L = len(sequence)
    pos1 = list(range(1, L + 1)) if positions_1based is None else list(positions_1based)
    for i, (wt, p1) in enumerate(zip(sequence, pos1)):
        for j, mut in enumerate(AMINO_ACIDS):
            if mut == wt:
                continue
            rows.append((p1, wt, mut, float(matrix[i, j])))
    return pd.DataFrame(rows, columns=["position", "wt", "mut", colname])
