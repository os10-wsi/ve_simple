"""Pipeline orchestration: fetch -> score -> store -> evaluate -> plot, for a list of UniProt IDs.

Stages run over the whole ID list so that each large model (ESM-1v x5, SaProt) is loaded once.
Every per-protein intermediate is cached under `cache/<accession>/` and re-used on re-runs.
"""
from __future__ import annotations

import logging
import os
import shutil
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import alphafold, cachecheck, clinvar, seqmap, uniprot
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
    foldx_workers: int = 1                             # parallel FoldX processes per protein
    foldx_scope: str = "all"                           # all | clinvar
    thermompnn_download: bool = True
    thermompnn_compute: bool = True
    n_boot: int = 1000
    min_per_class: int = 1
    include_likely: bool = True
    plot_predictors: list[str] = field(default_factory=lambda: list(DEFAULT_PLOT_PREDICTORS))
    make_pdf: bool = True
    cache_only: bool = False                           # compute + cache scores; write only proteins.tsv/summary.json
    am_bulk: bool = True                               # fill AlphaMissense gaps from the bulk release
    batch_proteins: int = 25                           # proteins processed together per stage
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


class ScorerPool:
    """Keeps ESM-1v / SaProt models loaded across protein batches.

    On a GPU all six models stay resident (about 8 GB in fp16); on CPU only one model is kept to
    bound memory, so models are reloaded once per batch.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._scorers: dict[str, object] = {}
        self._max_resident: Optional[int] = None

    def _limit(self) -> int:
        if self._max_resident is None:
            from .predictors.lm_common import resolve_device
            self._max_resident = 8 if resolve_device(self.cfg.device) == "cuda" else 1
        return self._max_resident

    def _evict_for(self, key: str) -> None:
        while len(self._scorers) >= self._limit() and key not in self._scorers:
            old_key, old = next(iter(self._scorers.items()))
            old.close()
            del self._scorers[old_key]

    def _load(self, key: str, factory):
        """Load a scorer; on CUDA out-of-memory, drop every resident model and try once more."""
        import torch
        if key in self._scorers:
            return self._scorers[key]
        self._evict_for(key)
        try:
            self._scorers[key] = factory()
        except torch.OutOfMemoryError:
            log.warning("CUDA out of memory while loading %s; unloading other models and retrying", key)
            self.close()
            torch.cuda.empty_cache()
            self._scorers[key] = factory()
        return self._scorers[key]

    def esm1v(self, model_name: str):
        from .predictors import esm1v as esm_mod
        return self._load(model_name, lambda: esm_mod.ESM1vScorer(model_name, device=self.cfg.device,
                                                                  batch_size=self.cfg.batch_size))

    def saprot(self):
        from .predictors import saprot as sp_mod
        return self._load("saprot", lambda: sp_mod.SaProtScorer(device=self.cfg.device, batch_size=self.cfg.batch_size))

    def close(self) -> None:
        for sc in self._scorers.values():
            sc.close()
        self._scorers.clear()


def _cache_path(cfg: Config, acc: str, name: str) -> Path:
    return Path(cfg.cache_dir) / acc / f"scores_{name}.tsv"


def _load_cached(cfg: Config, acc: str, name: str, sequence: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Read a cached score table; a broken file (truncated by a killed or duplicate job) is
    removed and treated as missing so that it is recomputed."""
    p = _cache_path(cfg, acc, name)
    if not p.exists() or cfg.force:
        return None
    problem = cachecheck.check_score_table(p, sequence)
    if problem:
        log.warning("%s: discarding cached %s (%s)", acc, p.name, problem)
        p.unlink()
        return None
    return pd.read_csv(p, sep="\t")


def _save_cache(cfg: Config, acc: str, name: str, df: pd.DataFrame) -> None:
    """Write atomically so that a concurrent reader never sees a partial file."""
    p = _cache_path(cfg, acc, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, p)


def remove_esm1v_matrices(acc_dir: Path) -> int:
    """Per-model ESM-1v matrices are only needed until the ensemble table is written."""
    n = 0
    for p in acc_dir.glob("esm1v_*.npy"):
        p.unlink(missing_ok=True)
        n += 1
    return n


def _load_npy(path: Path, sequence: str) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    problem = cachecheck.check_npy(path, sequence)
    if problem:
        log.warning("discarding cached %s (%s)", path, problem)
        path.unlink()
        return None
    return np.load(path)


def _save_npy(path: Path, mat: np.ndarray) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npy")
    np.save(tmp, mat)
    os.replace(tmp, path)


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


def _merge_mapped(st: ProteinState, df: Optional[pd.DataFrame], cols: list[str], name: str,
                  on=VARIANT_KEY) -> None:
    """Merge a table computed on an AlphaFold model or archive sequence: residue numbers are first
    checked against (and if needed aligned to) the current UniProt sequence."""
    if df is not None and len(df):
        df, frac = seqmap.remap_to_sequence(df, st.entry.sequence, f"{st.acc} {name}")
        if frac is not None:
            st.notes.setdefault("remapped", {})[name] = round(frac, 3)
    st.variants = _merge(st.variants, df, cols, on=on)


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
def stage_fetch(cfg: Config, ids: Optional[list[str]] = None) -> list[ProteinState]:
    states = []
    for acc in (cfg.ids if ids is None else ids):
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
                df = clinvar.load_clinvar(entry.accession, cfg.cache_dir)
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
        _merge_mapped(st, exp, ["solvent_exposure", "plddt"], "solvent_exposure", on=["position", "wt"])
        # AlphaMissense
        am = None
        if st.am_csv is not None and "alphamissense" not in cfg.skip:
            try:
                am = am_mod.read_alphamissense(st.am_csv)
            except Exception as e:  # noqa: BLE001
                st.errors.append(f"alphamissense: {e}")
        _merge_mapped(st, am, ["alphamissense", "alphamissense_class"], "alphamissense")


# ---------------------------------------------------------------------- stage 3: ThermoMPNN
def prepare_thermompnn_archive(cfg: Config) -> Optional[tm_mod.ZenodoThermoMPNN]:
    """Download/index the Zenodo archive and copy every requested protein's predictions straight
    into its cache table, in ONE streaming pass and without unpacking the archive."""
    if "thermompnn" in cfg.skip:
        return None
    zen = tm_mod.ZenodoThermoMPNN(cfg.data_dir)
    if zen.ensure_archive(download=cfg.thermompnn_download) is None:
        return None
    zen.build_index()
    todo = [acc for acc in cfg.ids if zen.available(acc) and _load_cached(cfg, acc, "thermompnn") is None]
    if todo:
        def sink(acc: str, df: pd.DataFrame) -> None:
            df["thermompnn_source"] = "zenodo:18381534"
            _save_cache(cfg, acc, "thermompnn", df)
        zen.stream(todo, sink)
    return zen


def stage_thermompnn(cfg: Config, states: list[ProteinState], zen: Optional[tm_mod.ZenodoThermoMPNN]) -> None:
    if "thermompnn" in cfg.skip:
        for st in states:
            df = _load_cached(cfg, st.acc, "thermompnn")
            st.notes["thermompnn_source"] = df["thermompnn_source"].iloc[0] if df is not None and len(df) else "skipped"
            _merge_mapped(st, df, ["thermompnn"], "thermompnn")
        return
    for st in states:
        df = _load_cached(cfg, st.acc, "thermompnn")
        if df is None and cfg.thermompnn_compute and st.pdb is not None:
            run_dir = Path(cfg.cache_dir) / st.acc / "thermompnn_run"
            try:
                df = tm_mod.run_thermompnn(st.pdb, run_dir, cfg.external_dir)
                if df is not None and len(df):
                    df["thermompnn_source"] = "computed"
                    _save_cache(cfg, st.acc, "thermompnn", df)
                    shutil.rmtree(run_dir, ignore_errors=True)
            except Exception as e:  # noqa: BLE001
                st.errors.append(f"thermompnn_run: {e}")
                log.error("%s: ThermoMPNN inference failed: %s", st.acc, e)
        st.notes["thermompnn_source"] = (df["thermompnn_source"].iloc[0]
                                         if df is not None and len(df) and "thermompnn_source" in df.columns
                                         else "missing")
        _merge_mapped(st, df, ["thermompnn"], "thermompnn")


# -------------------------------------------------------------------------- stage 4: ESM-1v
def stage_esm1v(cfg: Config, states: list[ProteinState], pool: ScorerPool) -> None:
    from .predictors import esm1v as esm_mod

    col = "esm1v"
    if col in cfg.skip:
        for st in states:
            st.variants = _merge(st.variants, _load_cached(cfg, st.acc, "esm1v", st.entry.sequence), [col, "esm1v_n_models"])
        return
    n_models = max(1, min(5, cfg.esm1v_models))
    todo = [st for st in states if _load_cached(cfg, st.acc, "esm1v", st.entry.sequence) is None]
    per_model: dict[str, list[np.ndarray]] = {st.acc: [] for st in todo}
    for mi, name in enumerate(esm_mod.ESM1V_MODELS[:n_models]):
        for st in todo:
            npy = Path(cfg.cache_dir) / st.acc / f"esm1v_{mi + 1}.npy"
            cached = None if cfg.force else _load_npy(npy, st.entry.sequence)
            if cached is not None:
                per_model[st.acc].append(cached)
                continue
            scorer = pool.esm1v(name)
            try:
                log.info("%s: ESM-1v model %d/%d (%d positions)", st.acc, mi + 1, n_models, len(st.entry.sequence))
                mat = scorer.score_matrix(st.entry.sequence)
                _save_npy(npy, mat)
                per_model[st.acc].append(mat)
            except Exception as e:  # noqa: BLE001
                st.errors.append(f"esm1v: {e}")
                log.error("%s: ESM-1v failed: %s\n%s", st.acc, e, traceback.format_exc())
    for st in states:
        df = _load_cached(cfg, st.acc, "esm1v", st.entry.sequence)
        if df is None and per_model.get(st.acc):
            if len(per_model[st.acc]) == n_models:
                df = esm_mod.esm1v_frame(per_model[st.acc], st.entry.sequence)
                _save_cache(cfg, st.acc, "esm1v", df)
                remove_esm1v_matrices(Path(cfg.cache_dir) / st.acc)
            else:
                # incomplete ensemble (a model failed): report nothing and keep the per-model
                # matrices so that a re-run only computes the missing member(s)
                log.error("%s: ESM-1v ensemble incomplete (%d/%d models); not cached",
                          st.acc, len(per_model[st.acc]), n_models)
        st.variants = _merge(st.variants, df, [col, "esm1v_n_models"])


# -------------------------------------------------------------------------- stage 5: SaProt
def stage_saprot(cfg: Config, states: list[ProteinState], pool: ScorerPool) -> None:

    col = "saprot"
    if col in cfg.skip:
        for st in states:
            _merge_mapped(st, _load_cached(cfg, st.acc, "saprot"), [col], "saprot")
        return
    for st in states:
        df = _load_cached(cfg, st.acc, "saprot")
        if df is None and st.pdb is not None:
            scorer = pool.saprot()
            try:
                log.info("%s: SaProt (%d positions)", st.acc, len(st.entry.sequence))
                df = scorer.score_frame(st.pdb)
                _save_cache(cfg, st.acc, "saprot", df)
            except Exception as e:  # noqa: BLE001
                st.errors.append(f"saprot: {e}")
                log.error("%s: SaProt failed: %s\n%s", st.acc, e, traceback.format_exc())
        _merge_mapped(st, df, [col], "saprot")


# --------------------------------------------------------------------------- stage 6: FoldX
def stage_foldx(cfg: Config, states: list[ProteinState]) -> None:
    col = "foldx"
    binary = foldx_mod.find_foldx(cfg.foldx_binary)
    if col in cfg.skip or binary is None:
        if col not in cfg.skip:
            log.warning("FoldX binary not found (set --foldx-binary or FOLDX_BINARY); foldx column left empty")
        for st in states:
            df = _load_cached(cfg, st.acc, "foldx")
            _merge_mapped(st, df, [col], "foldx")
        return
    for st in states:
        df = _load_cached(cfg, st.acc, "foldx")
        if df is None and st.pdb is not None:
            positions = None
            if cfg.foldx_scope == "clinvar":
                positions = sorted(set(st.clinvar_df["position"])) if len(st.clinvar_df) else []
            try:
                log.info("%s: FoldX BuildModel (%s positions)", st.acc, "all" if positions is None else len(positions))
                work = Path(cfg.cache_dir) / st.acc / "foldx"
                df = foldx_mod.foldx_saturation(binary, st.pdb, work, chunk_size=cfg.foldx_chunk,
                                                positions=positions, workers=cfg.foldx_workers)
                if df["foldx"].notna().all():
                    _save_cache(cfg, st.acc, "foldx", df)
                    shutil.rmtree(work, ignore_errors=True)
                else:
                    # keep finished chunks for the next run; failed chunks are retried then
                    n_bad = int(df["foldx"].isna().sum())
                    st.errors.append(f"foldx: {n_bad} mutants failed")
                    log.error("%s: FoldX failed for %d of %d mutants; not cached, rerun to retry",
                              st.acc, n_bad, len(df))
            except Exception as e:  # noqa: BLE001
                st.errors.append(f"foldx: {e}")
                log.error("%s: FoldX failed: %s", st.acc, e)
        _merge_mapped(st, df, [col], "foldx")


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
        "remapped_predictors": ";".join(f"{k}:{v}" for k, v in st.notes.get("remapped", {}).items()),
        "errors": "; ".join(st.errors),
    })
    row.update(clinvar.summarise(st.clinvar_df))
    for pred in PREDICTOR_DIRECTIONS:
        if pred in st.variants.columns:
            row[f"n_scored_{pred}"] = int(st.variants[pred].notna().sum())
    return row


def prepare_alphamissense(cfg: Config) -> None:
    """Fill AlphaMissense for human proteins that AlphaFold DB does not serve, in one bulk pass.

    Done only when building result tables (not in --cache-only chunks).  UniProt and AlphaFold DB
    metadata fetched here are cached, so the per-batch stages re-use them.
    """
    if cfg.cache_only or not cfg.am_bulk or "alphamissense" in cfg.skip:
        return
    missing = []
    for acc in cfg.ids:
        try:
            entry = uniprot.fetch_entry(acc, cfg.cache_dir)
            if entry is None or not entry.is_human:
                continue
            meta = alphafold.fetch_afdb_metadata(entry.accession, cfg.cache_dir)
            if alphafold.fetch_alphamissense(entry.accession, cfg.cache_dir, meta) is None:
                missing.append(entry.accession)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: AlphaMissense availability check failed: %s", acc, e)
    if missing:
        try:
            am_mod.fill_from_bulk(missing, cfg.cache_dir, Path(cfg.data_dir).parent / "alphamissense")
        except Exception as e:  # noqa: BLE001
            log.error("AlphaMissense bulk fill failed: %s", e)


def run_pipeline(cfg: Config) -> dict:
    """Process proteins in batches; stream per-variant results to disk so memory stays bounded."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db = None if cfg.cache_only else ResultsDB(out / "ve_simple.sqlite")
    zen = prepare_thermompnn_archive(cfg)
    prepare_alphamissense(cfg)
    pool = ScorerPool(cfg)

    tsv_path = out / "variant_scores.tsv"
    cv_path = out / "clinvar_variants.tsv"
    for pth in (tsv_path, cv_path):
        if pth.exists():
            pth.unlink()
    if cfg.cache_only:
        for pth in (out / "ve_simple.sqlite", out / "auc_long.tsv", out / "auc_per_protein.tsv"):
            pth.unlink(missing_ok=True)
    prot_rows: list[dict] = []
    auc_frames: list[pd.DataFrame] = []
    labelled_frames: list[pd.DataFrame] = []   # pathogenic/benign variants only, for the ROC plots
    n_variants = 0
    n_batches = max(1, (len(cfg.ids) + cfg.batch_proteins - 1) // cfg.batch_proteins)

    for bi in range(n_batches):
        batch_ids = cfg.ids[bi * cfg.batch_proteins:(bi + 1) * cfg.batch_proteins]
        log.info("Batch %d/%d (%d proteins)", bi + 1, n_batches, len(batch_ids))
        states = stage_fetch(cfg, batch_ids)
        if not states:
            continue
        stage_simple(cfg, states)
        stage_thermompnn(cfg, states, zen)
        stage_esm1v(cfg, states, pool)
        stage_saprot(cfg, states, pool)
        stage_foldx(cfg, states)
        for st in states:
            attach_clinvar(st)
            prow = protein_row(st, cfg)
            prot_rows.append(prow)
            if cfg.cache_only:
                continue
            db.write_protein(prow)
            db.write_variants(st.variants)
            st.variants.to_csv(tsv_path, sep="\t", index=False, mode="a", header=not tsv_path.exists())
            n_variants += len(st.variants)
            if len(st.clinvar_df):
                db.write_clinvar(st.clinvar_df)
                st.clinvar_df.to_csv(cv_path, sep="\t", index=False, mode="a", header=not cv_path.exists())
            if st.entry.is_human:
                auc = auc_for_protein(st.variants, st.acc, n_boot=cfg.n_boot, min_per_class=cfg.min_per_class)
                auc_frames.append(auc)
                db.write_auc(auc)
                lab = st.variants[st.variants["clinvar_label"].isin(["pathogenic", "benign"])]
                if len(lab):
                    labelled_frames.append(lab)
        del states
    pool.close()

    if not prot_rows:
        raise SystemExit("No valid UniProt accessions could be fetched.")
    if cfg.cache_only:
        proteins = pd.DataFrame(prot_rows)
        proteins.to_csv(out / "proteins.tsv", sep="\t", index=False)
        summary = {"n_requested": len(cfg.ids), "n_proteins": len(prot_rows), "cache_only": True,
                   "outputs": {"proteins": str(out / "proteins.tsv")}}
        write_json(summary, out / "summary.json")
        return summary
    if not cv_path.exists():
        pd.DataFrame(columns=clinvar.CLINVAR_COLUMNS).to_csv(cv_path, sep="\t", index=False)
    proteins = pd.DataFrame(prot_rows)
    auc_long = pd.concat(auc_frames, ignore_index=True) if auc_frames else pd.DataFrame()
    human = proteins[proteins["is_human"]] if len(proteins) else proteins
    auc_wide = wide_auc_table(auc_long, human) if len(human) else pd.DataFrame()
    proteins.to_csv(out / "proteins.tsv", sep="\t", index=False)
    auc_long.to_csv(out / "auc_long.tsv", sep="\t", index=False)
    auc_wide.to_csv(out / "auc_per_protein.tsv", sep="\t", index=False)
    pdf = None
    if cfg.make_pdf and labelled_frames:
        labelled = pd.concat(labelled_frames, ignore_index=True)
        pdf = roc_pdf(labelled, human, out / "roc_curves.pdf", predictors=cfg.plot_predictors)
    db.close()
    summary = {
        "n_requested": len(cfg.ids), "n_proteins": len(prot_rows), "n_human": int(len(human)),
        "n_variants": int(n_variants), "n_with_auc": int(auc_long["accession"].nunique()) if len(auc_long) else 0,
        "outputs": {
            "proteins": str(out / "proteins.tsv"), "variant_scores": str(tsv_path),
            "clinvar_variants": str(cv_path), "auc_long": str(out / "auc_long.tsv"),
            "auc_per_protein": str(out / "auc_per_protein.tsv"), "roc_pdf": str(pdf) if pdf else None,
            "sqlite": str(out / "ve_simple.sqlite"),
        },
    }
    write_json(summary, out / "summary.json")
    return summary


def run_foldx_only(cfg: Config) -> dict:
    """FoldX for each protein, written to the cache only (no result files): for cluster arrays."""
    n_done = n_failed = 0
    for i in range(0, len(cfg.ids), cfg.batch_proteins):
        states = stage_fetch(cfg, cfg.ids[i:i + cfg.batch_proteins])
        for st in states:
            st.variants = st.variants[["accession", "position", "wt", "mut"]]
        stage_foldx(cfg, states)
        for st in states:
            if any(e.startswith("foldx") for e in st.errors) or "foldx" not in st.variants or \
                    st.variants["foldx"].isna().all():
                n_failed += 1
            else:
                n_done += 1
    log.info("FoldX finished for %d protein(s), failed or skipped for %d", n_done, n_failed)
    return {"done": n_done, "failed": n_failed}
