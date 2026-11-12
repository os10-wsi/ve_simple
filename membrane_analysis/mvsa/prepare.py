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
    res["in_tm"] = False
    res["juxta"] = False
    if len(seg):
        idx = res.set_index("accession").index
        for acc, g in seg.groupby("accession"):
            m = idx == acc
            if not m.any():
                continue
            pos = res.loc[m, "position"].to_numpy()
            in_tm = np.zeros(len(pos), bool)
            near = np.zeros(len(pos), bool)
            for a, b in zip(g.start, g.end):
                in_tm |= (pos >= a) & (pos <= b)
                near |= (pos >= a - JUXTA) & (pos <= b + JUXTA)
            res.loc[m, "in_tm"] = in_tm
            res.loc[m, "juxta"] = near & ~in_tm
    cls = classes.set_index("accession")["class_annotation"]
    res["protein_class"] = res["accession"].map(cls)
    res["location"] = assign_location(res["protein_class"], res["in_tm"], res["juxta"], res["rsa"])
    return res


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
    return cat(labelled), cat(sample), pd.DataFrame(summaries)


def _process(df, ctx, cls, rng, sample_rate, labelled, sample, summaries):
    df = df.merge(ctx, on=["accession", "position"], how="left")
    df["protein_class"] = df["accession"].map(cls)
    df["region"] = np.where(df["in_tm"].fillna(False).astype(bool), "TM",
                            np.where(df["juxta"].fillna(False).astype(bool), "Juxtamembrane", "Non-TM"))
    lab = df["clinvar_class"].notna() & (df["clinvar_class"].astype(str) != "")
    labelled.append(df[lab])
    sample.append(df[rng.random(len(df)) < sample_rate])
    for _, g in df.groupby("accession", sort=False):
        summaries.extend(summarise_protein(g))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ve", type=Path, required=True, help="ve_simple assembly directory (e.g. ../results_human)")
    ap.add_argument("--shape", type=Path, required=True, help="shape_analysis results directory")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--sample-rate", type=float, default=0.01, help="fraction of all substitutions kept for plots")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    args.out.mkdir(parents=True, exist_ok=True)

    classes = pd.read_parquet(args.shape / "classes.parquet")
    ann = pd.read_parquet(args.shape / "annotations.parquet")
    status = pd.read_parquet(args.shape / "status.parquet")
    ve_prot = pd.read_csv(args.ve / "proteins.tsv", sep="\t")
    log.info("building residue context")
    residues = residue_context(args.shape, classes, ann)
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

    auc = pd.read_csv(args.ve / "auc_long.tsv", sep="\t")
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
    meta = {"ve_results": str(args.ve.resolve()), "shape_results": str(args.shape.resolve()),
            "sample_rate": args.sample_rate, "n_labelled": int(len(labelled)), "n_sample": int(len(sample)),
            "n_proteins_ve": int(prot.in_ve_results.sum()), "created": time.strftime("%Y-%m-%d %H:%M")}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    log.info("done: %s", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
