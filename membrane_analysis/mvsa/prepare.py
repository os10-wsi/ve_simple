"""Build the compact tables the notebooks read, from the ve_simple and shape_analysis results.

Run once on the cluster (it streams the full variant table, tens of GB):
    python -m mvsa.prepare --ve ../results_human --shape ../shape_analysis/results --out data

Outputs (all Parquet, in --out):
    proteins.parquet          one row per protein: class, ClinVar counts, coverage, disorder, structure flags
    labelled_variants.parquet every ClinVar-classified missense variant with all scores and residue context
    variant_sample.parquet    random sample of ALL possible substitutions (default 1%) with the same columns
    protein_scores.parquet    per protein x predictor x region: distribution summary of oriented scores
    protein_auc.parquet       per protein x predictor AUC with errors (ve_simple auc_long.tsv)
    shape_units.parquet       shape metrics per unit (default mask) for all proteins in the shape results
    residues.parquet          per residue: pLDDT, DSSP, burial depth, relative SASA, location
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .groups import DIRECTION, EXPOSED_RSA, JUXTA

log = logging.getLogger("mvsa.prepare")

MAX_ASA = {"ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0, "GLN": 225.0, "GLU": 223.0,
           "GLY": 104.0, "HIS": 224.0, "ILE": 197.0, "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0,
           "PRO": 159.0, "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0}
MEMBRANE_CLASSES = {"MULTIPASS", "SINGLE_PASS", "BETA_BARREL_MEMBRANE"}
SCORE_COLS = list(DIRECTION)
VARIANT_COLS = ["accession", "position", "wt", "mut", *SCORE_COLS, "plddt", "clinvar_class", "clinvar_label",
                "clinvar_review_status", "clinvar_ids"]


def residue_context(shape: Path, classes: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    """Per residue: pLDDT, DSSP, burial depth, relative SASA, TM / juxtamembrane flags and location."""
    res = pd.read_parquet(shape / "residues.parquet",
                          columns=["accession", "resnum", "resname", "plddt", "dssp", "sasa", "depth_mean",
                                   "in_chain_trimmed", "domain_id"])
    res["rsa"] = res["sasa"] / res["resname"].map(MAX_ASA)
    res = res.rename(columns={"resnum": "position", "plddt": "plddt_model"})
    # TM segments (helical or beta) and intramembrane segments from the UniProt annotation
    seg = []
    for r in ann.itertuples():
        for a, b in zip(r.tm_start, r.tm_end):
            seg.append((r.accession, int(a), int(b)))
        for a, b in zip(r.intramem_start, r.intramem_end):
            seg.append((r.accession, int(a), int(b)))
    seg = pd.DataFrame(seg, columns=["accession", "start", "end"])
    res = res.reset_index(drop=True)
    in_tm, juxta = _tm_flags(res["accession"].to_numpy(), res["position"].to_numpy(), seg)
    res["in_tm"] = in_tm
    res["juxta"] = juxta
    cls = classes.set_index("accession")["class_annotation"]
    res["protein_class"] = res["accession"].map(cls)
    res["location"] = assign_location(res["protein_class"], res["in_tm"], res["juxta"], res["rsa"])
    return res


def _tm_flags(acc: np.ndarray, pos: np.ndarray, seg: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """In-TM and juxtamembrane flags per residue.  Residues are grouped by accession once, so each
    protein's segments touch only its own rows (linear in the number of residues)."""
    in_tm = np.zeros(len(pos), bool)
    near = np.zeros(len(pos), bool)
    if not len(seg) or not len(pos):
        return in_tm, near
    codes, uniq = pd.factorize(acc)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    lookup = {a: i for i, a in enumerate(uniq)}
    for a, g in seg.groupby("accession"):
        k = lookup.get(a)
        if k is None:
            continue
        rows = order[bounds[k]:bounds[k + 1]]
        p = pos[rows]
        t = np.zeros(len(p), bool)
        n = np.zeros(len(p), bool)
        for s0, e0 in zip(g["start"].to_numpy(), g["end"].to_numpy()):
            t |= (p >= s0) & (p <= e0)
            n |= (p >= s0 - JUXTA) & (p <= e0 + JUXTA)
        in_tm[rows] = t
        near[rows] = n
    return in_tm, near & ~in_tm


def assign_location(protein_class: pd.Series, in_tm: pd.Series, juxta: pd.Series, rsa: pd.Series) -> pd.Series:
    exposed = rsa >= EXPOSED_RSA
    memb = protein_class.isin(MEMBRANE_CLASSES)
    loc = np.where(exposed, "Exposed", "Buried")
    loc = np.where(memb & in_tm & exposed, "TM lipid-facing", loc)
    loc = np.where(memb & in_tm & ~exposed, "TM buried", loc)
    loc = np.where(memb & ~in_tm & juxta, "Juxtamembrane", loc)
    loc = np.where(memb & ~in_tm & ~juxta & exposed, "Extramembrane exposed", loc)
    loc = np.where(memb & ~in_tm & ~juxta & ~exposed, "Extramembrane buried", loc)
    loc = np.where(rsa.isna(), "Unknown", loc)
    return pd.Series(loc, index=protein_class.index)


def summarise_protein(g: pd.DataFrame, region_col: str = "region") -> list[dict]:
    """Distribution summary of oriented scores for one protein, overall and by TM / non-TM region."""
    rows = []
    for region, sub in [("all", g)] + list(g.groupby(region_col)):
        for p, sign in DIRECTION.items():
            if p not in sub:
                continue
            v = sub[p].to_numpy(float) * sign
            ok = np.isfinite(v)
            if ok.sum() < 19:
                continue
            v = v[ok]
            q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
            pos_means = sub.loc[ok, ["position"]].assign(v=v).groupby("position")["v"]
            rows.append({"accession": g["accession"].iat[0], "region": region, "predictor": p, "n": int(ok.sum()),
                         "mean": float(v.mean()), "sd": float(v.std()), "q05": q[0], "q25": q[1], "median": q[2],
                         "q75": q[3], "q95": q[4], "dynamic_range": q[4] - q[0],
                         "between_position_sd": float(pos_means.mean().std()),
                         "within_position_sd": float(pos_means.std().mean())})
    return rows


def stream_variants(path: Path, residues: pd.DataFrame, classes: pd.DataFrame, sample_rate: float,
                    chunksize: int, seed: int = 0):
    """One pass over variant_scores.tsv: labelled variants, a random sample, per-protein summaries."""
    rng = np.random.default_rng(seed)
    header = pd.read_csv(path, sep="\t", nrows=0).columns
    cols = [c for c in VARIANT_COLS if c in header]
    ctx = residues[["accession", "position", "rsa", "depth_mean", "dssp", "plddt_model", "in_tm", "juxta", "location",
                    "in_chain_trimmed"]]
    cls = classes.set_index("accession")["class_annotation"]
    labelled, sample, summaries = [], [], []
    carry = None
    n_rows = 0
    t0 = time.time()
    reader = pd.read_csv(path, sep="\t", usecols=cols, chunksize=chunksize,
                         dtype={"clinvar_class": "object", "clinvar_label": "object", "clinvar_review_status": "object",
                                "clinvar_ids": "object"})
    for chunk in reader:
        n_rows += len(chunk)
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last = chunk["accession"].iat[-1]
        carry = chunk[chunk["accession"] == last]
        done = chunk[chunk["accession"] != last]
        if len(done):
            _process(done, ctx, cls, rng, sample_rate, labelled, sample, summaries)
        log.info("%.1f M rows read (%.0f s)", n_rows / 1e6, time.time() - t0)
    if carry is not None and len(carry):
        _process(carry, ctx, cls, rng, sample_rate, labelled, sample, summaries)
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    return cat(labelled), cat(sample), cat(summaries)


QUANTS = {"q05": 0.05, "q25": 0.25, "median": 0.5, "q75": 0.75, "q95": 0.95}


def summarise_chunk(df: pd.DataFrame, region_col: str = "region") -> pd.DataFrame:
    """Vectorised summarise_protein over every protein in df (same numbers).  Groups are integer
    codes computed once per chunk; grouping on strings per predictor was the slow part."""
    out = []
    acc_code, acc_uniq = pd.factorize(df["accession"], sort=False)
    reg_code, reg_uniq = pd.factorize(df[region_col].astype(str), sort=False)
    pos = df["position"].to_numpy(np.int64)
    span = int(pos.max()) + 1 if len(pos) else 1
    nreg = len(reg_uniq) + 1                       # last region code means "all"
    keysets = {"all": acc_code.astype(np.int64) * nreg + (nreg - 1),
               "region": acc_code.astype(np.int64) * nreg + reg_code}
    regions = np.append(np.asarray(reg_uniq, dtype=object), "all")
    for p, sign in DIRECTION.items():
        if p not in df:
            continue
        v = pd.to_numeric(df[p], errors="coerce").to_numpy(float) * sign
        ok = np.isfinite(v)
        if not ok.any():
            continue
        for key in keysets.values():
            gid = key[ok]
            d = pd.DataFrame({"g": gid, "gp": gid * span + pos[ok], "v": v[ok]})
            g = d.groupby("g", sort=False)["v"]
            st = pd.DataFrame({"n": g.size(), "mean": g.mean(), "sd": g.std(ddof=0)})
            st = st[st["n"] >= 19]
            if not len(st):
                continue
            q = g.quantile(list(QUANTS.values())).unstack()
            q.columns = list(QUANTS)
            pg = d.groupby("gp", sort=False)["v"]
            pstat = pd.DataFrame({"m": pg.mean(), "s": pg.std()})
            pstat["g"] = pstat.index.to_numpy() // span
            st = st.join(q).join(pstat.groupby("g")["m"].std().rename("between_position_sd")) \
                   .join(pstat.groupby("g")["s"].mean().rename("within_position_sd"))
            st["dynamic_range"] = st["q95"] - st["q05"]
            gi = st.index.to_numpy()
            st.insert(0, "accession", np.asarray(acc_uniq, dtype=object)[gi // nreg])
            st.insert(1, "region", regions[gi % nreg])
            st["predictor"] = p
            out.append(st.reset_index(drop=True))
    cols = ["accession", "region", "predictor", "n", "mean", "sd", *QUANTS, "dynamic_range",
            "between_position_sd", "within_position_sd"]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def _process(df, ctx, cls, rng, sample_rate, labelled, sample, summaries):
    df = df.merge(ctx, on=["accession", "position"], how="left")
    df["protein_class"] = df["accession"].map(cls)
    df["region"] = np.where(df["in_tm"].fillna(False).astype(bool), "TM",
                            np.where(df["juxta"].fillna(False).astype(bool), "Juxtamembrane", "Non-TM"))
    lab = df["clinvar_class"].notna() & (df["clinvar_class"].astype(str) != "")
    labelled.append(df[lab])
    sample.append(df[rng.random(len(df)) < sample_rate])
    summaries.append(summarise_chunk(df))


def check_inputs(ve: Path, shape: Path) -> None:
    """Stop early, with a clear message, if an upstream step has not finished."""
    missing_ve = [f for f in ("variant_scores.tsv", "proteins.tsv", "auc_long.tsv") if not (ve / f).exists()]
    if missing_ve:
        raise SystemExit(f"{ve} is missing {', '.join(missing_ve)}: the ve_simple assembly (lsf/phase3_assemble.sh) "
                         "has not finished for this proteome.")
    missing_shape = [f for f in ("classes.parquet", "annotations.parquet", "metrics.parquet", "residues.parquet",
                                 "status.parquet") if not (shape / f).exists()]
    if missing_shape:
        raise SystemExit(f"{shape} is missing {', '.join(missing_shape)}: run `protshape collect` and "
                         "`protshape classify` in shape_analysis first.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ve", type=Path, required=True, help="ve_simple assembly directory (e.g. ../results_human)")
    ap.add_argument("--shape", type=Path, required=True, help="shape_analysis results directory")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--sample-rate", type=float, default=0.01, help="fraction of all substitutions kept for plots")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--allow-partial", action="store_true",
                    help="proceed even if the shape results cover less than half of the ve_simple proteins")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    check_inputs(args.ve, args.shape)
    args.out.mkdir(parents=True, exist_ok=True)

    classes = pd.read_parquet(args.shape / "classes.parquet")
    ann = pd.read_parquet(args.shape / "annotations.parquet")
    status = pd.read_parquet(args.shape / "status.parquet")
    ve_prot = pd.read_csv(args.ve / "proteins.tsv", sep="\t")
    log.info("building residue context")
    residues = residue_context(args.shape, classes, ann)
    covered = ve_prot["accession"].isin(set(residues["accession"])).mean()
    log.info("shape results cover %.0f%% of the %d proteins in %s", 100 * covered, len(ve_prot), args.ve)
    if covered < 0.5 and not args.allow_partial:
        raise SystemExit(
            f"The shape results in {args.shape} cover only {100 * covered:.0f}% of the proteins in {args.ve}. "
            "The full shape run has probably not been collected yet (protshape collect). Finish it first, or pass "
            "--allow-partial to continue with structural context for the covered proteins only.")
    # residue numbering of the AlphaFold model only matches the variants when the sequences agree
    mismatch = set(ve_prot.loc[ve_prot.get("structure_sequence_mismatch", False).astype(bool), "accession"])
    residues.loc[residues.accession.isin(mismatch), ["rsa", "depth_mean", "location"]] = [np.nan, np.nan, "Unknown"]
    residues.to_parquet(args.out / "residues.parquet", index=False)

    log.info("streaming %s", args.ve / "variant_scores.tsv")
    labelled, sample, summaries = stream_variants(args.ve / "variant_scores.tsv", residues, classes,
                                                  args.sample_rate, args.chunksize)
    labelled.to_parquet(args.out / "labelled_variants.parquet", index=False)
    sample.to_parquet(args.out / "variant_sample.parquet", index=False)
    summaries.to_parquet(args.out / "protein_scores.parquet", index=False)

    try:
        auc = pd.read_csv(args.ve / "auc_long.tsv", sep="\t")
    except pd.errors.EmptyDataError:            # no ClinVar labels (e.g. yeast): no per-protein AUCs
        auc = pd.DataFrame(columns=["accession", "predictor", "n_pathogenic", "n_benign", "n_scored", "auc",
                                    "auc_se", "ci_low", "ci_high"])
    auc.to_parquet(args.out / "protein_auc.parquet", index=False)

    keep = ["accession", "taxid", "protein_class", "unit", "mask_variant", "domain_id", "domain_label", "cath_class",
            "n_res"]
    m = pd.read_parquet(args.shape / "metrics.parquet")
    m = m[m.mask_variant == "default"]
    num = [c for c in m.columns if c not in keep and pd.api.types.is_numeric_dtype(m[c])]
    m[[c for c in keep if c in m.columns] + num].to_parquet(args.out / "shape_units.parquet", index=False)

    prot = classes.merge(ann[["accession", "n_tm_helical", "n_tm_beta", "n_intramem", "n_lipid", "subcellular_location"]],
                         on="accession", how="left")
    prot = prot.merge(status[["accession", "frac_excluded", "in_ted_frac", "ted_domains"]], on="accession", how="left")
    vcols = [c for c in ("accession", "n_clinvar_missense", "n_pathogenic", "n_benign", "n_vus", "n_conflicting",
                         "has_alphamissense", "thermompnn_source", "remapped_predictors", "structure_sequence_mismatch",
                         "is_fragment") if c in ve_prot.columns]
    vcols += [c for c in ve_prot.columns if c.startswith("n_scored_")]
    prot = prot.merge(ve_prot[vcols], on="accession", how="left")
    plddt = residues.groupby("accession")["plddt_model"].mean().rename("plddt_mean")
    prot = prot.merge(plddt, left_on="accession", right_index=True, how="left")
    prot["in_ve_results"] = prot["accession"].isin(set(ve_prot["accession"]))
    prot.to_parquet(args.out / "proteins.parquet", index=False)
    taxa = prot.loc[prot.in_ve_results, "taxid"]
    meta = {"ve_results": str(args.ve.resolve()), "shape_results": str(args.shape.resolve()),
            "taxid": int(taxa.mode().iat[0]) if len(taxa) else None,
            "n_labelled_pathogenic_benign": int(labelled["clinvar_class"].isin(
                ["pathogenic", "likely_pathogenic", "benign", "likely_benign"]).sum()) if len(labelled) else 0,
            "sample_rate": args.sample_rate, "n_labelled": int(len(labelled)), "n_sample": int(len(sample)),
            "n_proteins_ve": int(prot.in_ve_results.sum()), "created": time.strftime("%Y-%m-%d %H:%M")}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    log.info("done: %s", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
