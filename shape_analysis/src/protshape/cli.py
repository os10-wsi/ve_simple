"""protshape command line: index-cache, annotate, classify, select-stage0, compute, collect, sanity, analyze."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import METRICS_VERSION
from .cache_index import index_cache, read_id_file

log = logging.getLogger("protshape")


def _ids(args) -> list[str] | None:
    ids = []
    for f in args.ids or []:
        ids += read_id_file(f)
    ids += list(getattr(args, "id", None) or [])
    return list(dict.fromkeys(ids)) or None


def cmd_index(args) -> int:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    idx = index_cache(Path(args.cache), threads=args.threads)
    idx.to_parquet(work / "index.parquet", index=False)
    print(f"Indexed {len(idx)} cached proteins in {args.cache}")
    print(idx.groupby(["taxid", "organism"]).agg(n=("accession", "size"), with_model=("has_model", "sum"),
                                               fragments=("is_fragment", "sum")).to_string())
    print("AFDB model versions:", idx.loc[idx.has_model, "model_version"].value_counts().to_dict())
    for f in args.ids or []:
        want = read_id_file(f)
        have = set(idx.loc[idx.has_model, "accession"])
        missing = [a for a in want if a not in have]
        tag = Path(f).stem
        (work / f"missing_{tag}.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
        print(f"{tag}: {len(want)} accessions listed, {len(want) - len(missing)} with a cached model, "
              f"{len(missing)} missing (written to {work / f'missing_{tag}.txt'})")
    return 0


def cmd_annotate(args) -> int:
    from .annotations import annotate
    from .ted import fetch_ted

    work = Path(args.work)
    idx = pd.read_parquet(work / "index.parquet")
    ann = annotate(idx, threads=args.threads)
    ann.to_parquet(work / "annotations.parquet", index=False)
    print(f"UniProt annotations for {len(ann)} proteins -> {work / 'annotations.parquet'}")
    if not args.no_ted:
        fetch_ted(idx.loc[idx.has_model, "accession"].tolist(), work / "ted", threads=args.threads)
    return 0


def cmd_classify(args) -> int:
    from .classify import add_final_class, classify

    work = Path(args.work)
    cls = classify(pd.read_parquet(work / "annotations.parquet"))
    mpath = work / "metrics.parquet"
    if mpath.exists():
        m = pd.read_parquet(mpath, columns=["accession", "unit", "mask_variant", "belt_width_scan_A_kd",
                                            "belt_contrast_flanked_kd"])
        cls = add_final_class(cls, m)
        print("Confusion (annotation class x belt detected on CHAIN_TRIMMED):")
        print(pd.crosstab(cls["class_annotation"], cls["belt_detected"].fillna("no unit")).to_string())
    cls.to_parquet(work / "classes.parquet", index=False)
    print(cls.groupby(["taxid", "class_annotation"]).size().unstack(fill_value=0).to_string())
    return 0


def cmd_select_stage0(args) -> int:
    work = Path(args.work)
    cls = pd.read_parquet(work / "classes.parquet")
    idx = pd.read_parquet(work / "index.parquet")[["accession", "has_model"]]
    cls = cls.merge(idx, on="accession")
    cls = cls[cls.has_model]
    rng = np.random.default_rng(args.seed)
    picks = []
    taxa = [9606, 559292]                                   # human and yeast
    classes = ("MULTIPASS", "SOLUBLE")
    per = args.n // (len(classes) * len(taxa))
    for c in classes:
        for t in taxa:
            pool = cls[(cls.class_annotation == c) & (cls.taxid == t)]["accession"].tolist()
            picks += list(rng.choice(pool, size=min(per, len(pool)), replace=False))
    out = work / "stage0_ids.txt"
    out.write_text("\n".join(picks) + "\n")
    print(f"{len(picks)} accessions -> {out}")
    return 0


def _jobs(work: Path, accessions: list[str] | None, variants, max_box: int, use_final=False) -> list[dict]:
    from .ted import load_domains, load_queried

    idx = pd.read_parquet(work / "index.parquet").set_index("accession")
    ann = pd.read_parquet(work / "annotations.parquet").set_index("accession")
    cls = pd.read_parquet(work / "classes.parquet").set_index("accession")
    ted = load_domains(work / "ted")
    ted_by = {a: g.to_dict("records") for a, g in ted.groupby("accession")}
    queried = set(load_queried(work / "ted"))
    accs = accessions or idx.index[idx.has_model].tolist()
    jobs, no_ted = [], []
    for a in accs:
        if a not in queried:          # TED query failed or not run: do not mistake it for "no domains"
            no_ted.append(a)
            continue
        if a not in idx.index or not idx.at[a, "has_model"] or a not in ann.index:
            log.warning("%s: no cached model or annotation; skipped", a)
            continue
        jobs.append({
            "accession": a, "model_path": idx.at[a, "model_path"], "model_version": int(idx.at[a, "model_version"]),
            "taxid": int(idx.at[a, "taxid"]), "is_fragment": bool(idx.at[a, "is_fragment"]),
            "ann": ann.loc[a].to_dict(), "protein_class": cls.at[a, "class_annotation"] if a in cls.index else "AMBIGUOUS",
            "ted": ted_by.get(a, []), "variants": variants, "max_box": max_box,
        })
    if no_ted:
        log.warning("%d proteins skipped because their TED query has not succeeded yet (e.g. %s); "
                    "run `protshape annotate` again, then compute", len(no_ted), ", ".join(no_ted[:5]))
    return jobs


def cmd_compute(args) -> int:
    from .compute import load_done, run_compute

    work = Path(args.work)
    variants = tuple(args.variants.split(","))
    jobs = _jobs(work, _ids(args), variants, args.max_box)
    if args.nshards > 1:
        jobs = jobs[args.shard::args.nshards]
    done = load_done(work / "compute", {j["accession"]: j["model_version"] for j in jobs})
    todo = [j for j in jobs if j["accession"] not in done]
    print(f"{len(jobs)} proteins in scope, {len(done)} already computed, {len(todo)} to compute")
    if todo:
        res = run_compute(todo, work / "compute", args.workers, tag=f"s{args.shard:04d}")
        print(f"computed {res['done']} proteins ({res['errors']} errors) in {res['seconds']} s")
    return 0


def cmd_collect(args) -> int:
    from .classify import add_final_class
    from .compute import collect

    work = Path(args.work)
    parts = collect(work / "compute")
    m, st = parts["metrics"], parts["status"]
    if m.empty:
        print("no computed metrics found")
        return 1
    st = st[st.metrics_version == METRICS_VERSION]
    m = m[m.metrics_version == METRICS_VERSION]           # ignore rows computed by older versions
    ok = st[st.status == "ok"][["accession", "model_version"]]
    m = m.merge(ok, on=["accession", "model_version"]).drop_duplicates(
        ["accession", "unit", "mask_variant", "domain_id"], keep="last")
    cls = pd.read_parquet(work / "classes.parquet")
    cls = add_final_class(cls, m)
    cls.to_parquet(work / "classes.parquet", index=False)
    m = m.drop(columns=[c for c in ("class_annotation", "class_final") if c in m.columns]).merge(
        cls[["accession", "class_annotation", "class_final", "belt_detected"]], on="accession", how="left")
    m["organism"] = m["taxid"].map({9606: "human", 559292: "yeast"}).fillna(m["taxid"].astype(str))
    m.to_parquet(work / "metrics.parquet", index=False)
    parts["residues"].to_parquet(work / "residues.parquet", index=False)
    st.to_parquet(work / "status.parquet", index=False)
    print(f"{len(m)} unit rows for {m.accession.nunique()} proteins -> {work / 'metrics.parquet'}")
    print(m.groupby(["unit", "mask_variant"]).size().to_string())
    errs = st[st.status != "ok"]
    if len(errs):
        print(f"{len(errs)} proteins failed; see status.parquet")
    return 0


def cmd_sanity(args) -> int:
    from .sanity import sanity_table

    work = Path(args.work)
    tab = sanity_table(pd.read_parquet(work / "metrics.parquet"))
    tab.to_csv(work / "sanity.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tab.to_string(index=False))
    print(f"\nwritten to {work / 'sanity.tsv'}")
    return 0 if (tab["check"] != "FAIL").all() else 1


def cmd_analyze(args) -> int:
    from .analysis import run_analysis

    run_analysis(Path(args.work), Path(args.work) / "analysis")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="protshape", description=__doc__)
    p.add_argument("--work", default="results", help="working/results directory (default: results)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("index-cache", help="index the ve_simple cache by accession")
    s.add_argument("--cache", default="../cache")
    s.add_argument("--ids", nargs="*", type=Path, help="proteome ID files to check coverage against")
    s.add_argument("--threads", type=int, default=16)
    s.set_defaults(func=cmd_index)
    s = sub.add_parser("annotate", help="UniProt annotations from the cache, and TED domains from ted.cathdb.info")
    s.add_argument("--threads", type=int, default=4, help="parallel TED queries (TED rate-limits above ~4)")
    s.add_argument("--no-ted", action="store_true")
    s.set_defaults(func=cmd_annotate)
    s = sub.add_parser("classify", help="membrane/soluble classes (class_final once metrics exist)")
    s.set_defaults(func=cmd_classify)
    s = sub.add_parser("select-stage0", help="pick N proteins (half MULTIPASS, half SOLUBLE, split by organism)")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_select_stage0)
    s = sub.add_parser("compute", help="compute all units and metrics (resumable, shardable)")
    s.add_argument("--ids", nargs="*", type=Path)
    s.add_argument("--id", nargs="*")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--nshards", type=int, default=1)
    s.add_argument("--variants", default="default,plddt50,plddt90,no_ted",
                   help="mask variants for CHAIN_TRIMMED/DOMAIN (sensitivity analysis needs all four)")
    s.add_argument("--max-box", type=int, default=256, help="maximum voxel box size per side (1 A voxels)")
    s.set_defaults(func=cmd_compute)
    s = sub.add_parser("collect", help="merge computed parts into metrics.parquet / residues.parquet")
    s.set_defaults(func=cmd_collect)
    s = sub.add_parser("sanity", help="print the stage-4 sanity table")
    s.set_defaults(func=cmd_sanity)
    s = sub.add_parser("analyze", help="stage-5 analyses, figures and findings.md")
    s.set_defaults(func=cmd_analyze)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
