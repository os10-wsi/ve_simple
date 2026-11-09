"""Command-line interface: `ve-simple run --ids ids.txt --out results/`."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .constants import PREDICTOR_DIRECTIONS
from .utils import normalise_accession, read_id_list


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ids", type=Path, help="text file with UniProt accessions (one per line or whitespace/comma separated)")
    p.add_argument("--id", dest="id_list", nargs="*", default=[], help="UniProt accessions given directly")
    p.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    p.add_argument("--cache", type=Path, default=Path("cache"), help="per-protein cache directory")
    p.add_argument("--data-dir", type=Path, default=Path("data/zenodo"), help="where the Zenodo ThermoMPNN archive lives")
    p.add_argument("--external-dir", type=Path, default=Path("external"), help="where external code (ThermoMPNN) is cloned")
    p.add_argument("--device", default="auto", help="torch device for ESM-1v/SaProt: auto|cpu|cuda|mps")
    p.add_argument("--batch-size", type=int, default=None, help="masked positions per forward pass")
    p.add_argument("--esm1v-models", type=int, default=5, help="number of ESM-1v ensemble members (1-5)")
    p.add_argument("--skip", default="",
                   help="comma-separated predictors not to compute (cached scores from earlier runs are still used), "
                        "e.g. esm1v,saprot,foldx")
    p.add_argument("--foldx-binary", default=None, help="path to the FoldX executable (or set FOLDX_BINARY)")
    p.add_argument("--foldx-scope", choices=["all", "clinvar"], default="all",
                   help="run FoldX on every position (default) or only positions with ClinVar variants")
    p.add_argument("--foldx-chunk", type=int, default=200, help="mutants per FoldX BuildModel call")
    p.add_argument("--foldx-workers", type=int, default=1,
                   help="FoldX processes to run in parallel per protein (FoldX itself is single-threaded)")
    p.add_argument("--no-thermompnn-download", action="store_true", help="do not download the Zenodo archive")
    p.add_argument("--no-thermompnn-compute", action="store_true", help="never run ThermoMPNN de novo")
    p.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for AUC confidence intervals")
    p.add_argument("--min-per-class", type=int, default=1, help="minimum pathogenic and benign variants to compute an AUC")
    p.add_argument("--strict-labels", action="store_true",
                   help="use only Pathogenic/Benign (exclude Likely pathogenic / Likely benign)")
    p.add_argument("--plot-predictors", default=None, help="comma-separated predictors to draw in the ROC PDF")
    p.add_argument("--batch-proteins", type=int, default=25,
                   help="proteins processed together per stage (bounds memory; models stay loaded across batches)")
    p.add_argument("--no-pdf", action="store_true", help="skip the ROC PDF (useful for proteome-scale runs)")
    p.add_argument("--cache-only", action="store_true",
                   help="compute and cache scores but write only proteins.tsv and summary.json "
                        "(for cluster chunks; the full tables are built afterwards by the assembly step)")
    p.add_argument("--force", action="store_true", help="recompute cached predictor scores")
    p.add_argument("-v", "--verbose", action="store_true")


def _set_hf_home(cache_dir: Path) -> None:
    """Keep model weights (16 GB for ESM-1v x5 + SaProt) next to the pipeline cache, not in $HOME."""
    import os

    if not os.environ.get("HF_HOME"):
        hf = Path(cache_dir).resolve() / "huggingface"
        hf.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf)
    logging.getLogger("ve_simple").info("Hugging Face weights cache: %s", os.environ["HF_HOME"])


def cmd_run(args: argparse.Namespace) -> int:
    _set_hf_home(args.cache)
    from .pipeline import Config, run_pipeline

    ids = read_id_list(args.ids) if args.ids else []
    ids += [normalise_accession(a) for a in args.id_list]
    ids = list(dict.fromkeys(ids))
    if not ids:
        print("No UniProt IDs given (use --ids FILE or --id ACC ...)", file=sys.stderr)
        return 2
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    unknown = skip - set(PREDICTOR_DIRECTIONS)
    if unknown:
        print(f"Unknown predictor(s) in --skip: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    cfg = Config(
        ids=ids, out_dir=args.out, cache_dir=args.cache, data_dir=args.data_dir, external_dir=args.external_dir,
        device=args.device, esm1v_models=args.esm1v_models, batch_size=args.batch_size, skip=skip,
        foldx_binary=args.foldx_binary, foldx_chunk=args.foldx_chunk, foldx_workers=args.foldx_workers, foldx_scope=args.foldx_scope,
        thermompnn_download=not args.no_thermompnn_download, thermompnn_compute=not args.no_thermompnn_compute,
        n_boot=args.bootstrap, min_per_class=args.min_per_class, include_likely=not args.strict_labels,
        force=args.force, batch_proteins=max(1, args.batch_proteins), make_pdf=not args.no_pdf,
        cache_only=args.cache_only,
    )
    if args.plot_predictors:
        cfg.plot_predictors = [s.strip() for s in args.plot_predictors.split(",") if s.strip()]
    summary = run_pipeline(cfg)
    print("Done.")
    for k, v in summary["outputs"].items():
        if v:
            print(f"  {k:16s} {v}")
    return 0


def cmd_fetch_zenodo(args: argparse.Namespace) -> int:
    from .predictors.thermompnn import ZenodoThermoMPNN

    zen = ZenodoThermoMPNN(args.data_dir)
    zen.ensure_archive(download=True)
    idx = zen.build_index(force=args.reindex)
    print(f"Archive: {zen.archive}\nProteins indexed: {len(idx)}")
    if args.extract_all:
        print("--extract-all is no longer needed (the archive is read directly) and is ignored; "
              f"if {zen.extract_dir} exists you can delete it.")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Re-run AUC statistics and plots from an existing results database."""
    import pandas as pd

    from .db import ResultsDB
    from .evaluate import auc_for_protein, wide_auc_table
    from .plots import DEFAULT_PLOT_PREDICTORS, roc_pdf

    db = ResultsDB(Path(args.out) / "ve_simple.sqlite")
    proteins = db.read("proteins")
    if proteins.empty:
        print("No results found in", args.out, file=sys.stderr)
        return 1
    human = proteins[proteins["is_human"].astype(bool)]
    # only labelled variants are needed for AUCs and plots; read them once instead of the whole table
    labelled = pd.read_sql("SELECT * FROM variants WHERE clinvar_label IN ('pathogenic','benign')", db.conn)
    frames = []
    for acc in human["accession"]:
        sub = labelled[labelled["accession"] == acc]
        auc = auc_for_protein(sub, acc, n_boot=args.bootstrap, min_per_class=args.min_per_class)
        frames.append(auc)
        db.write_auc(auc)
    auc_long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    auc_long.to_csv(Path(args.out) / "auc_long.tsv", sep="\t", index=False)
    wide_auc_table(auc_long, human).to_csv(Path(args.out) / "auc_per_protein.tsv", sep="\t", index=False)
    preds = [s.strip() for s in args.plot_predictors.split(",")] if args.plot_predictors else DEFAULT_PLOT_PREDICTORS
    if not args.no_pdf and len(labelled):
        roc_pdf(labelled, human, Path(args.out) / "roc_curves.pdf", preds)
    db.close()
    print("Updated AUC tables and ROC PDF in", args.out)
    return 0


def cmd_check_cache(args: argparse.Namespace) -> int:
    from .cachecheck import check_cache

    accs = read_id_list(args.ids) if args.ids else None
    problems = check_cache(args.cache, delete=args.delete, accessions=accs)
    by_acc = {}
    for path, _ in problems:
        by_acc.setdefault(path.parent.name, 0)
        by_acc[path.parent.name] += 1
    print(f"{len(problems)} problem file(s) in {len(by_acc)} accession(s)" + (" deleted" if args.delete else
          " found; re-run with --delete to remove them so they are recomputed"))
    return 0


def cmd_foldx(args: argparse.Namespace) -> int:
    from .pipeline import Config, run_foldx_only
    from .predictors.foldx import find_foldx

    ids = read_id_list(args.ids) if args.ids else []
    ids += [normalise_accession(a) for a in args.id_list]
    ids = list(dict.fromkeys(ids))
    if not ids:
        print("No UniProt IDs given (use --ids FILE or --id ACC ...)", file=sys.stderr)
        return 2
    if find_foldx(args.foldx_binary) is None:
        print("FoldX binary not found: pass --foldx-binary or set FOLDX_BINARY", file=sys.stderr)
        return 2
    cfg = Config(ids=ids, cache_dir=args.cache, foldx_binary=args.foldx_binary, foldx_chunk=args.foldx_chunk,
                 foldx_workers=args.foldx_workers, foldx_scope=args.foldx_scope, batch_proteins=10)
    res = run_foldx_only(cfg)
    return 0 if res["failed"] == 0 else 1


def cmd_compact_cache(args: argparse.Namespace) -> int:
    from .cachecheck import compact_cache

    accs = read_id_list(args.ids) if args.ids else None
    counts = compact_cache(args.cache, data_dir=args.data_dir, accessions=accs, dry_run=args.dry_run,
                           remove_zenodo_extracted=args.remove_zenodo_extracted)
    verb = "would remove" if args.dry_run else "removed"
    print(f"{verb} {counts['files_removed']} file(s); converted {counts['clinvar_converted']} ClinVar record(s); "
          f"{counts['thermompnn_reset']} multi-fragment ThermoMPNN table(s) reset for rebuilding")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ve-simple", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="score a list of UniProt IDs and benchmark against ClinVar")
    _add_run_args(p_run)
    p_run.set_defaults(func=cmd_run)
    p_z = sub.add_parser("fetch-zenodo", help="download and index the proteome-wide ThermoMPNN archive")
    p_z.add_argument("--data-dir", type=Path, default=Path("data/zenodo"))
    p_z.add_argument("--reindex", action="store_true")
    p_z.add_argument("--extract-all", action="store_true", help="ignored (kept for old instructions)")
    p_z.add_argument("-v", "--verbose", action="store_true")
    p_z.set_defaults(func=cmd_fetch_zenodo)
    p_e = sub.add_parser("evaluate", help="recompute AUC tables and ROC plots from an existing results directory")
    p_e.add_argument("--out", type=Path, default=Path("results"))
    p_e.add_argument("--bootstrap", type=int, default=1000)
    p_e.add_argument("--min-per-class", type=int, default=1)
    p_e.add_argument("--plot-predictors", default=None)
    p_e.add_argument("--no-pdf", action="store_true")
    p_e.add_argument("-v", "--verbose", action="store_true")
    p_e.set_defaults(func=cmd_evaluate)
    p_c = sub.add_parser("check-cache", help="find (and with --delete remove) truncated or unreadable cache files")
    p_c.add_argument("--cache", type=Path, default=Path("cache"))
    p_c.add_argument("--ids", type=Path, default=None, help="restrict to these accessions (default: whole cache)")
    p_c.add_argument("--delete", action="store_true")
    p_c.add_argument("-v", "--verbose", action="store_true")
    p_c.set_defaults(func=cmd_check_cache)
    p_f = sub.add_parser("foldx", help="run FoldX only, writing to the cache (for cluster job arrays)")
    p_f.add_argument("--ids", type=Path)
    p_f.add_argument("--id", dest="id_list", nargs="*", default=[])
    p_f.add_argument("--cache", type=Path, default=Path("cache"))
    p_f.add_argument("--foldx-binary", default=None)
    p_f.add_argument("--foldx-scope", choices=["all", "clinvar"], default="all")
    p_f.add_argument("--foldx-chunk", type=int, default=200)
    p_f.add_argument("--foldx-workers", type=int, default=1)
    p_f.add_argument("-v", "--verbose", action="store_true")
    p_f.set_defaults(func=cmd_foldx)
    p_k = sub.add_parser("compact-cache", help="delete redundant cache files to reduce the file count")
    p_k.add_argument("--cache", type=Path, default=Path("cache"))
    p_k.add_argument("--data-dir", type=Path, default=Path("data/zenodo"))
    p_k.add_argument("--ids", type=Path, default=None, help="restrict to these accessions (default: whole cache)")
    p_k.add_argument("--remove-zenodo-extracted", action="store_true",
                     help="also delete the unpacked archive under <data-dir>/thermompnn_outs (~20,000 files)")
    p_k.add_argument("--dry-run", action="store_true")
    p_k.add_argument("-v", "--verbose", action="store_true")
    p_k.set_defaults(func=cmd_compact_cache)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
