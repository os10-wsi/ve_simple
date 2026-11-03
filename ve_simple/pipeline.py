"""Pipeline orchestration: fetch -> score -> store -> evaluate -> plot, for a list of UniProt IDs.

Stages run over the whole ID list so that each large model (ESM-1v x5, SaProt) is loaded once.
Every per-protein intermediate is cached under `cache/<accession>/` and re-used on re-runs.
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import alphafold, clinvar, uniprot
from .constants import PREDICTOR_DIRECTIONS
from .db import ResultsDB
from .evaluate import auc_for_protein, wide_auc_table
from .plots import DEFAULT_PLOT_PREDICTORS, roc_pdf
from .predictors import alphamissense as am_mod
from .predictors import foldx as foldx_mod
from .predictors import thermompnn as tm_mod
from .predictors.exposure import solvent_exposure
from .predictors.physchem import physchem_scores, saturation_table
from .uniprot import ProteinEntry
from .utils import write_json

log = logging.getLogger(__name__)

VARIANT_KEY = ["position", "wt", "mut"]


@dataclass
class Config:
    ids: list[str]
    out_dir: Path = Path("results")
    cache_dir: Path = Path("cache")
    data_dir: Path = Path("data/zenodo")
    external_dir: Path = Path("external")
    device: str = "auto"
    esm1v_models: int = 5
    batch_size: Optional[int] = None
    skip: set[str] = field(default_factory=set)       # predictor names to skip
    foldx_binary: Optional[str] = None
    foldx_chunk: int = 200
    foldx_scope: str = "all"                           # all | clinvar
    thermompnn_download: bool = True
    thermompnn_compute: bool = True
    n_boot: int = 1000
    min_per_class: int = 1
    include_likely: bool = True
    plot_predictors: list[str] = field(default_factory=lambda: list(DEFAULT_PLOT_PREDICTORS))
    force: bool = False


@dataclass
class ProteinState:
    entry: ProteinEntry
    afdb: Optional[dict] = None
    pdb: Optional[Path] = None
    am_csv: Optional[Path] = None
    clinvar_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    variants: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def acc(self) -> str:
        return self.entry.accession


def _cache_path(cfg: Config, acc: str, name: str) -> Path:
    return Path(cfg.cache_dir) / acc / f"scores_{name}.tsv"


def _load_cached(cfg: Config, acc: str, name: str) -> Optional[pd.DataFrame]:
    p = _cache_path(cfg, acc, name)
    if p.exists() and not cfg.force:
        return pd.read_csv(p, sep="\t")
    return None


def _save_cache(cfg: Config, acc: str, name: str, df: pd.DataFrame) -> None:
    p = _cache_path(cfg, acc, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, sep="\t", index=False)


def _merge(variants: pd.DataFrame, df: Optional[pd.DataFrame], cols: list[str], on=VARIANT_KEY) -> pd.DataFrame:
    """Left-join predictor columns onto the saturation table, validating wild-type residues."""
    for c in cols:
        if c in variants.columns:
            variants = variants.drop(columns=c)
    if df is None or len(df) == 0:
        for c in cols:
            variants[c] = np.nan
        return variants
    df = df[on + [c for c in cols if c in df.columns]].drop_duplicates(on)
    return variants.merge(df, on=on, how="left")


def _check_structure_matches(st: ProteinState) -> bool:
    """AlphaFold models are built from the UniProt sequence; verify before mapping positions."""
    if st.pdb is None:
        return False
    _, seq, nums, _ = alphafold.read_pdb_chain(st.pdb)
    ok = seq == st.entry.sequence and list(nums) == list(range(1, len(seq) + 1))
    if not ok:
        st.notes["structure_sequence_mismatch"] = True
        log.warning("%s: AlphaFold model sequence differs from UniProt sequence; structural "
                    "predictors will be aligned by residue number only", st.acc)
    return ok


# ------------------------------------------------------------------------------- stage 1: fetch
def stage_fetch(cfg: Config) -> list[ProteinState]:
    states = []
    for acc in cfg.ids:
        try:
            entry = uniprot.fetch_entry(acc, cfg.cache_dir)
        except Exception as e:  # noqa: BLE001
            log.error("%s: UniProt fetch failed: %s", acc, e)
            continue
        if entry is None:
            continue
        st = ProteinState(entry=entry)
        try:
            st.afdb = alphafold.fetch_afdb_metadata(entry.accession, cfg.cache_dir)
            st.pdb = alphafold.fetch_structure(entry.accession, cfg.cache_dir, st.afdb)
            if st.pdb is None:
                st.notes["no_alphafold_model"] = True
                log.warning("%s: no AlphaFold model; structure-based predictors skipped", st.acc)
            else:
                _check_structure_matches(st)
        except Exception as e:  # noqa: BLE001
            st.errors.append(f"alphafold: {e}")
            log.error("%s: AlphaFold fetch failed: %s", st.acc, e)
        if entry.is_human:
            try:
                st.am_csv = alphafold.fetch_alphamissense(entry.accession, cfg.cache_dir, st.afdb)
            except Exception as e:  # noqa: BLE001
                st.errors.append(f"alphamissense: {e}")
            try:
                data = clinvar.fetch_variation(entry.accession, cfg.cache_dir)
                df = clinvar.parse_clinvar_missense(entry.accession, data)
                df["clinvar_label"] = [clinvar.class_to_label(c, cfg.include_likely) for c in df["clinvar_class"]]
                st.clinvar_df = df
            except Exception as e:  # noqa: BLE001
                st.errors.append(f"clinvar: {e}")
                log.error("%s: ClinVar fetch failed: %s", st.acc, e)
        st.variants = saturation_table(entry.sequence)
        st.variants.insert(0, "accession", entry.accession)
        states.append(st)
        log.info("%s (%s, %s, %d aa): human=%s, structure=%s, ClinVar missense=%d",
                 st.acc, entry.gene, entry.organism, len(entry.sequence), entry.is_human,
                 st.pdb is not None, len(st.clinvar_df))
    return states


# ------------------------------------------------------------------- stage 2: cheap predictors
def stage_simple(cfg: Config, states: list[ProteinState]) -> None:
    for st in states:
        st.variants = physchem_scores(st.variants)
        # solvent exposure + pLDDT (per position)
        exp = _load_cached(cfg, st.acc, "exposure")
        if exp is None and st.pdb is not None and "solvent_exposure" not in cfg.skip:
            try:
                exp = solvent_exposure(st.pdb)
                _save_cache(cfg, st.acc, "exposure", exp)
            except Exception as e:  # noqa: BLE001
                st.errors.append(f"solvent_exposure: {e}")
                log.error("%s: SASA failed: %s", st.acc, e)
        st.variants = _merge(st.variants, exp, ["solvent_exposure", "plddt"], on=["position", "wt"])
        # AlphaMissense
        am = None
        if st.am_csv is not None and "alphamissense" not in cfg.skip:
            try:
                am = am_mod.read_alphamissense(st.am_csv)
            except Exception as e:  # noqa: BLE001
                st.errors.append(f"alphamissense: {e}")
        st.variants = _merge(st.variants, am, ["alphamissense", "alphamissense_class"])


# ---------------------------------------------------------------------- stage 3: ThermoMPNN
def stage_thermompnn(cfg: Config, states: list[ProteinState]) -> None:
    if "thermompnn" in cfg.skip:
        for st in states:
            st.variants = _merge(st.variants, None, ["thermompnn"])
        return
    zen = tm_mod.ZenodoThermoMPNN(cfg.data_dir)
    archive = zen.ensure_archive(download=cfg.thermompnn_download)
    todo = [st for st in states if _load_cached(cfg, st.acc, "thermompnn") is None]
    if archive is not None and todo:
        zen.build_index()
        zen.extract([st.acc for st in todo if st.entry.is_human])
    for st in states:
        df = _load_cached(cfg, st.acc, "thermompnn")
        source = st.notes.get("thermompnn_source")
        if df is None and archive is not None and zen.available(st.acc):
            try:
                df = zen.load(st.acc, st.entry.sequence)
                source = "zenodo:18381534"
            except Exception as e:  # noqa: BLE001
                st.errors.append(f"thermompnn_zenodo: {e}")
                log.error("%s: could not read Zenodo ThermoMPNN file: %s", st.acc, e)
        if df is None and cfg.thermompnn_compute and st.pdb is not None:
            try:
                df = tm_mod.run_thermompnn(st.pdb, Path(cfg.cache_dir) / st.acc / "thermompnn_run", cfg.external_dir)
                source = "computed"
            except Exception as e:  # noqa: BLE001
                st.errors.append(f"thermompnn_run: {e}")
                log.error("%s: ThermoMPNN inference failed: %s", st.acc, e)
        if df is not None and len(df):
            if "thermompnn_source" not in df.columns:
                df["thermompnn_source"] = source
            _save_cache(cfg, st.acc, "thermompnn", df)
            st.notes["thermompnn_source"] = df["thermompnn_source"].iloc[0]
        else:
            st.notes["thermompnn_source"] = "missing"
        st.variants = _merge(st.variants, df, ["thermompnn"])


# -------------------------------------------------------------------------- stage 4: ESM-1v
def stage_esm1v(cfg: Config, states: list[ProteinState]) -> None:
    from .predictors import esm1v as esm_mod

    col = "esm1v"
    if col in cfg.skip:
        for st in states:
            st.variants = _merge(st.variants, None, [col])
        return
    n_models = max(1, min(5, cfg.esm1v_models))
    todo = [st for st in states if _load_cached(cfg, st.acc, "esm1v") is None]
    per_model: dict[str, list[np.ndarray]] = {st.acc: [] for st in todo}
    for mi, name in enumerate(esm_mod.ESM1V_MODELS[:n_models]):
        pending = [st for st in todo if not (Path(cfg.cache_dir) / st.acc / f"esm1v_{mi + 1}.npy").exists() or cfg.force]
        scorer = None
        for st in todo:
            npy = Path(cfg.cache_dir) / st.acc / f"esm1v_{mi + 1}.npy"
            if npy.exists() and not cfg.force:
                per_model[st.acc].append(np.load(npy))
                continue
            if scorer is None:
                scorer = esm_mod.ESM1vScorer(name, device=cfg.device, batch_size=cfg.batch_size)
            try:
                log.info("%s: ESM-1v model %d/%d (%d positions)", st.acc, mi + 1, n_models, len(st.entry.sequence))
                mat = scorer.score_matrix(st.entry.sequence)
                np.save(npy, mat)
                per_model[st.acc].append(mat)
            except Exception as e:  # noqa: BLE001
                st.errors.append(f"esm1v: {e}")
                log.error("%s: ESM-1v failed: %s\n%s", st.acc, e, traceback.format_exc())
        if scorer is not None:
            scorer.close()
        del pending
    for st in states:
        df = _load_cached(cfg, st.acc, "esm1v")
        if df is None and per_model.get(st.acc):
            df = esm_mod.esm1v_frame(per_model[st.acc], st.entry.sequence)
            _save_cache(cfg, st.acc, "esm1v", df)
        st.variants = _merge(st.variants, df, [col, "esm1v_n_models"])


# -------------------------------------------------------------------------- stage 5: SaProt
def stage_saprot(cfg: Config, states: list[ProteinState]) -> None:
    from .predictors import saprot as sp_mod

    col = "saprot"
    if col in cfg.skip:
        for st in states:
            st.variants = _merge(st.variants, None, [col])
        return
    scorer = None
    for st in states:
        df = _load_cached(cfg, st.acc, "saprot")
        if df is None and st.pdb is not None:
            if scorer is None:
                scorer = sp_mod.SaProtScorer(device=cfg.device, batch_size=cfg.batch_size)
            try:
                log.info("%s: SaProt (%d positions)", st.acc, len(st.entry.sequence))
                df = scorer.score_frame(st.pdb)
                _save_cache(cfg, st.acc, "saprot", df)
            except Exception as e:  # noqa: BLE001
                st.errors.append(f"saprot: {e}")
                log.error("%s: SaProt failed: %s\n%s", st.acc, e, traceback.format_exc())
        st.variants = _merge(st.variants, df, [col])
    if scorer is not None:
        scorer.close()


# --------------------------------------------------------------------------- stage 6: FoldX
def stage_foldx(cfg: Config, states: list[ProteinState]) -> None:
    col = "foldx"
    binary = foldx_mod.find_foldx(cfg.foldx_binary)
    if col in cfg.skip or binary is None:
        if col not in cfg.skip:
            log.warning("FoldX binary not found (set --foldx-binary or FOLDX_BINARY); foldx column left empty")
        for st in states:
            df = _load_cached(cfg, st.acc, "foldx")
            st.variants = _merge(st.variants, df, [col])
        return
    for st in states:
        df = _load_cached(cfg, st.acc, "foldx")
        if df is None and st.pdb is not None:
            positions = None
            if cfg.foldx_scope == "clinvar":
                positions = sorted(set(st.clinvar_df["position"])) if len(st.clinvar_df) else []
            try:
                log.info("%s: FoldX BuildModel (%s positions)", st.acc, "all" if positions is None else len(positions))
                df = foldx_mod.foldx_saturation(binary, st.pdb, Path(cfg.cache_dir) / st.acc / "foldx",
                                                chunk_size=cfg.foldx_chunk, positions=positions)
                _save_cache(cfg, st.acc, "foldx", df)
            except Exception as e:  # noqa: BLE001
                st.errors.append(f"foldx: {e}")
                log.error("%s: FoldX failed: %s", st.acc, e)
        st.variants = _merge(st.variants, df, [col])


# ------------------------------------------------------------- stage 7: labels, DB, evaluation
def attach_clinvar(st: ProteinState) -> None:
    cols = ["clinvar_class", "clinvar_label", "clinvar_significances", "clinvar_review_status", "clinvar_ids"]
    v = st.variants.drop(columns=[c for c in cols if c in st.variants.columns])
    if len(st.clinvar_df):
        cv = st.clinvar_df[VARIANT_KEY + cols]
        # ClinVar variants whose wild-type residue does not match the UniProt sequence are dropped
        v = v.merge(cv, on=VARIANT_KEY, how="left")
        n_unmatched = len(cv.merge(st.variants[VARIANT_KEY], on=VARIANT_KEY, how="left", indicator=True)
                          .query("_merge == 'left_only'"))
        if n_unmatched:
            st.notes["clinvar_unmatched_wt"] = n_unmatched
    else:
        for c in cols:
            v[c] = np.nan
    v["clinvar_label"] = v["clinvar_label"].fillna("")
    st.variants = v


def protein_row(st: ProteinState, cfg: Config) -> dict:
    row = st.entry.to_dict()
    row.update({
        "alphafold_model": st.afdb.get("modelEntityId") if st.afdb else None,
        "alphafold_version": st.afdb.get("latestVersion") if st.afdb else None,
        "has_structure": st.pdb is not None,
        "has_alphamissense": st.am_csv is not None,
        "thermompnn_source": st.notes.get("thermompnn_source"),
        "structure_sequence_mismatch": bool(st.notes.get("structure_sequence_mismatch", False)),
        "errors": "; ".join(st.errors),
    })
    row.update(clinvar.summarise(st.clinvar_df))
    for pred in PREDICTOR_DIRECTIONS:
        if pred in st.variants.columns:
            row[f"n_scored_{pred}"] = int(st.variants[pred].notna().sum())
    return row


def run_pipeline(cfg: Config) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db = ResultsDB(out / "ve_simple.sqlite")

    states = stage_fetch(cfg)
    if not states:
        raise SystemExit("No valid UniProt accessions could be fetched.")
    stage_simple(cfg, states)
    stage_thermompnn(cfg, states)
    stage_esm1v(cfg, states)
    stage_saprot(cfg, states)
    stage_foldx(cfg, states)

    prot_rows, auc_frames = [], []
    for st in states:
        attach_clinvar(st)
        prow = protein_row(st, cfg)
        prot_rows.append(prow)
        db.write_protein(prow)
        db.write_variants(st.variants)
        if len(st.clinvar_df):
            db.write_clinvar(st.clinvar_df)
        if st.entry.is_human:
            auc = auc_for_protein(st.variants, st.acc, n_boot=cfg.n_boot, min_per_class=cfg.min_per_class)
            auc_frames.append(auc)
            db.write_auc(auc)
        write_json({"notes": st.notes, "errors": st.errors}, Path(cfg.cache_dir) / st.acc / "status.json")

    proteins = pd.DataFrame(prot_rows)
    variants = pd.concat([st.variants for st in states], ignore_index=True)
    auc_long = pd.concat(auc_frames, ignore_index=True) if auc_frames else pd.DataFrame()
    human = proteins[proteins["is_human"]] if len(proteins) else proteins
    auc_wide = wide_auc_table(auc_long, human) if len(human) else pd.DataFrame()

    proteins.to_csv(out / "proteins.tsv", sep="\t", index=False)
    variants.to_csv(out / "variant_scores.tsv", sep="\t", index=False)
    cv_all = pd.concat([st.clinvar_df for st in states if len(st.clinvar_df)], ignore_index=True) \
        if any(len(st.clinvar_df) for st in states) else pd.DataFrame(columns=clinvar.CLINVAR_COLUMNS)
    cv_all.to_csv(out / "clinvar_variants.tsv", sep="\t", index=False)
    auc_long.to_csv(out / "auc_long.tsv", sep="\t", index=False)
    auc_wide.to_csv(out / "auc_per_protein.tsv", sep="\t", index=False)
    pdf = None
    if len(human):
        hv = variants[variants["accession"].isin(human["accession"])]
        pdf = roc_pdf(hv, human, out / "roc_curves.pdf", predictors=cfg.plot_predictors)
    db.close()
    summary = {
        "n_requested": len(cfg.ids), "n_proteins": len(states), "n_human": int(len(human)),
        "n_variants": int(len(variants)), "n_with_auc": int(auc_long["accession"].nunique()) if len(auc_long) else 0,
        "outputs": {
            "proteins": str(out / "proteins.tsv"), "variant_scores": str(out / "variant_scores.tsv"),
            "clinvar_variants": str(out / "clinvar_variants.tsv"), "auc_long": str(out / "auc_long.tsv"),
            "auc_per_protein": str(out / "auc_per_protein.tsv"), "roc_pdf": str(pdf) if pdf else None,
            "sqlite": str(out / "ve_simple.sqlite"),
        },
    }
    write_json(summary, out / "summary.json")
    return summary
