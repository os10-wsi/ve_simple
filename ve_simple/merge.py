"""Combine per-chunk result directories (from a parallel assembly) into one results directory."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

from . import clinvar
from .evaluate import wide_auc_table
from .plots import roc_pdf
from .utils import write_json

log = logging.getLogger(__name__)

LABELLED = "labelled_variant_scores.tsv"   # pathogenic/benign rows only, written by every non-cache run


def part_status(parts_dir: Path) -> tuple[list[Path], list[Path]]:
    """(finished, unfinished) part directories; a part is finished when its summary.json exists."""
    parts = sorted(p for p in Path(parts_dir).iterdir() if p.is_dir())
    done = [p for p in parts if (p / "summary.json").exists()]
    return done, [p for p in parts if p not in done]


def _concat_tsv(files: list[Path], dest: Path) -> int:
    """Concatenate TSVs with one header.  Files whose header matches the first are copied byte for
    byte (fast); any other file is re-aligned to the first file's columns.  Returns data rows."""
    files = [f for f in files if f.exists() and f.stat().st_size > 0]
    tmp = dest.with_name(dest.name + ".tmp")
    n = 0
    with open(tmp, "wb") as out:
        header: Optional[bytes] = None
        cols: list[str] = []
        for f in files:
            with open(f, "rb") as fh:
                h = fh.readline()
                if header is None:
                    header, cols = h, h.decode().rstrip("\r\n").split("\t")
                    out.write(h)
                if h == header:
                    shutil.copyfileobj(fh, out, length=16 << 20)
                    continue
            log.warning("%s: columns differ from the first part; re-aligning", f)
            for chunk in pd.read_csv(f, sep="\t", chunksize=1_000_000, dtype=str, keep_default_na=False):
                extra = set(chunk.columns) - set(cols)
                if extra:
                    log.warning("%s: dropping columns absent from the first part: %s", f, sorted(extra))
                chunk.reindex(columns=cols).to_csv(out, sep="\t", index=False, header=False)
    tmp.replace(dest)
    if files:
        with open(dest, "rb") as fh:
            n = sum(buf.count(b"\n") for buf in iter(lambda: fh.read(16 << 20), b"")) - 1
    return max(n, 0)


def merge_parts(parts_dir: Path, out: Path, make_pdf: bool = True,
                plot_predictors: Optional[list[str]] = None, allow_partial: bool = False) -> dict:
    parts_dir, out = Path(parts_dir), Path(out)
    done, unfinished = part_status(parts_dir)
    if not done:
        raise SystemExit(f"No finished parts under {parts_dir}")
    if unfinished:
        msg = f"{len(unfinished)} of {len(done) + len(unfinished)} parts unfinished: " + \
              ", ".join(p.name for p in unfinished[:20])
        if not allow_partial:
            raise SystemExit(msg + "\nResubmit them (same phase3 command) or pass --allow-partial.")
        log.warning(msg)
    out.mkdir(parents=True, exist_ok=True)
    log.info("merging %d parts from %s into %s", len(done), parts_dir, out)

    def read_all(name: str) -> pd.DataFrame:
        frames = [pd.read_csv(p / name, sep="\t", low_memory=False) for p in done
                  if (p / name).exists() and (p / name).stat().st_size > 1]
        frames = [f for f in frames if len(f.columns)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    proteins = read_all("proteins.tsv")
    proteins.to_csv(out / "proteins.tsv", sep="\t", index=False)
    auc_long = read_all("auc_long.tsv")
    auc_long.to_csv(out / "auc_long.tsv", sep="\t", index=False)
    human = proteins[proteins["is_human"].astype(bool)] if len(proteins) else proteins
    (wide_auc_table(auc_long, human) if len(human) else pd.DataFrame()).to_csv(
        out / "auc_per_protein.tsv", sep="\t", index=False)
    labelled = read_all(LABELLED)
    if len(labelled):
        labelled.to_csv(out / LABELLED, sep="\t", index=False)

    n_cv = _concat_tsv([p / "clinvar_variants.tsv" for p in done], out / "clinvar_variants.tsv")
    if not (out / "clinvar_variants.tsv").stat().st_size:
        pd.DataFrame(columns=clinvar.CLINVAR_COLUMNS).to_csv(out / "clinvar_variants.tsv", sep="\t", index=False)
    log.info("concatenating variant_scores.tsv (the big one)")
    n_var = _concat_tsv([p / "variant_scores.tsv" for p in done], out / "variant_scores.tsv")

    pdf = None
    if make_pdf and len(labelled):
        pdf = roc_pdf(labelled, human, out / "roc_curves.pdf", predictors=plot_predictors)
    summary = {
        "n_parts": len(done), "n_parts_unfinished": len(unfinished),
        "unfinished_parts": [p.name for p in unfinished],
        "n_proteins": int(len(proteins)), "n_human": int(len(human)), "n_variants": int(n_var),
        "n_clinvar_rows": int(n_cv),
        "n_with_auc": int(auc_long["accession"].nunique()) if len(auc_long) else 0,
        "requested": sum(json.loads((p / "summary.json").read_text()).get("n_requested", 0) for p in done),
        "outputs": {k: str(out / v) for k, v in {
            "proteins": "proteins.tsv", "variant_scores": "variant_scores.tsv",
            "clinvar_variants": "clinvar_variants.tsv", "auc_long": "auc_long.tsv",
            "auc_per_protein": "auc_per_protein.tsv", "labelled_variant_scores": LABELLED}.items()},
    }
    summary["outputs"]["roc_pdf"] = str(pdf) if pdf else None
    write_json(summary, out / "summary.json")
    return summary
