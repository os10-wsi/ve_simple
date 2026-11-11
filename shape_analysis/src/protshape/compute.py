"""Per-protein computation of all units and metrics, parallel and resumable.

Output (under <out>/parts/), written in shards so the file count stays small:
    metrics-<tag>.parquet   one row per protein x unit x mask_variant (x domain)
    residues-<tag>.parquet  per-residue table (pLDDT, DSSP, SASA, burial depth, masks) for each protein
    status-<tag>.parquet    one row per protein: ok / error, n_units, timings, TED verification
A protein is skipped when a status row with the same model_version and metrics_version exists.
"""
from __future__ import annotations

import logging
import os
import time
import traceback
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import METRICS_VERSION
from . import geometry as G
from .scales import KYTE_DOOLITTLE, WIMLEY_WHITE_OCT, atom_scale_values
from .structure import Structure, read_pdb
from .units import MASK_VARIANTS, build_units, dssp, runs
from .voxel import BoxTooLarge, voxel_metrics

log = logging.getLogger(__name__)


def helix_stats(ss: np.ndarray, resnum: np.ndarray) -> dict:
    n = len(ss)
    hel = [b - a for a, b in runs(ss == "H", resnum) if b - a >= 4]
    return {
        "frac_helix": float((ss == "H").mean()) if n else np.nan,
        "frac_strand": float((ss == "E").mean()) if n else np.nan,
        "frac_coil": float((ss == "C").mean()) if n else np.nan,
        "n_helices": len(hel), "mean_helix_len": float(np.mean(hel)) if hel else 0.0,
        "longest_helix": int(max(hel)) if hel else 0, "n_helices_gt30": int(sum(h > 30 for h in hel)),
        "n_strand_segments": len([1 for a, b in runs(ss == "E", resnum) if b - a >= 2]),
    }


def unit_metrics(s: Structure, ss: np.ndarray, max_box: int) -> tuple[dict, Optional[np.ndarray], np.ndarray]:
    x, radii, masses = s.coords, s.radii, s.masses
    ca = x[s.ca_index[s.ca_index >= 0]]
    n = s.n_res
    m: dict = {"n_res": n, "n_atoms": len(x), "mass_heavy": float(masses.sum()), "plddt_mean": float(s.plddt.mean())}
    m["rg_mass"] = G.radius_of_gyration(x, masses)
    m["rg_ca"] = G.radius_of_gyration(ca) if len(ca) else np.nan
    m["rmin_empirical"] = G.rmin_empirical(n)
    m["rg_ca_over_rmin_empirical"] = m["rg_ca"] / m["rmin_empirical"]
    m["rh_proxy"] = m["rg_mass"] / 0.775
    m.update(G.gyration_descriptors(x))
    if len(ca) >= 3:
        m.update(G.gyration_descriptors(ca, prefix="ca_"))
    m.update(G.hull_metrics(x, radii))
    m["dmax_over_2rg"] = m["dmax"] / (2 * m["rg_mass"]) if m["rg_mass"] else np.nan
    m["contact_density_ca"] = G.contact_density(ca)
    atom_sasa = G.sasa(x, radii)
    m["sasa"] = float(atom_sasa.sum())
    m["sasa_scaled"] = m["sasa"] / n ** (2 / 3)
    m.update(helix_stats(ss, s.resnum))

    # residue-level hydrophobic moment
    for name, scale in (("kd", KYTE_DOOLITTLE), ("ww", WIMLEY_WHITE_OCT)):
        if len(ca) == n and n >= 3:
            h = np.array([scale.get(r, 0.0) for r in s.res_names])
            d = ca - ca.mean(0)
            u = d / np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
            mu = (h[:, None] * u).sum(0)
            m[f"hmoment_{name}"] = float(np.linalg.norm(mu) / n)
            pa = np.array([m["pa_x"], m["pa_y"], m["pa_z"]])
            cos = abs(mu @ pa) / max(np.linalg.norm(mu), 1e-9)
            m[f"hmoment_angle_{name}"] = float(np.degrees(np.arccos(np.clip(cos, 0, 1))))

    depth = None
    _, vecs, _ = G.gyration_tensor(x)
    try:
        vm, per_atom = voxel_metrics(x, radii, atom_scale_values(s.resname, s.atom_name, s.element),
                                     s.res_index, vecs, max_box=max_box)
        m.update(vm)
        depth = per_atom["depth"]
        m["voxel_status"] = "ok"
        m["rmin_sphere"] = G.rmin_sphere(m["v_vdw"])
        m["rg_over_rmin_sphere"] = m["rg_mass"] / m["rmin_sphere"]
        m["solidity"] = m["v_vdw"] / m["v_hull"] if m.get("v_hull") else np.nan
        m["solidity_ses"] = m["v_ses"] / m["v_hull"] if m.get("v_hull") else np.nan
        m["sphericity_ses"] = G.sphericity(m["v_ses"], m["a_ses"])
    except BoxTooLarge as e:
        m["voxel_status"] = f"box_too_large: {e}"
    return m, depth, atom_sasa


def process_protein(job: dict) -> dict:
    """Compute every unit for one protein. Never raises: errors are returned in the status row."""
    acc = job["accession"]
    t0 = time.time()
    status = {"accession": acc, "model_version": job["model_version"], "metrics_version": METRICS_VERSION,
              "status": "ok", "error": "", "n_units": 0}
    rows, residues = [], None
    try:
        s = read_pdb(Path(job["model_path"]))
        ss = dssp(Path(job["model_path"]), s.n_res)
        units, info = build_units(s, job["ann"], job["protein_class"], job["ted"], ss, variants=job["variants"])
        status.update(info)
        base = {"accession": acc, "taxid": job["taxid"], "model_version": job["model_version"],
                "metrics_version": METRICS_VERSION, "protein_class": job["protein_class"],
                "is_fragment": job["is_fragment"], "protein_n_res": s.n_res}
        res_depth = None
        res_sasa = None
        for u in units:
            sub = s.subset(u.mask)
            if sub.n_res < 3:
                continue
            row = {**base, "unit": u.unit, "mask_variant": u.mask_variant, "domain_id": u.domain_id, **u.meta}
            try:
                mtr, depth, atom_sasa = unit_metrics(sub, ss[u.mask], job["max_box"])
                row.update(mtr)
                if u.unit == "CHAIN_FULL":
                    res_sasa = np.bincount(sub.res_index, weights=atom_sasa, minlength=sub.n_res)
                    if depth is not None:
                        res_depth = (np.bincount(sub.res_index, weights=depth, minlength=sub.n_res)
                                     / np.bincount(sub.res_index, minlength=sub.n_res),
                                     np.array([depth[sub.res_index == i].max() for i in range(sub.n_res)]))
            except Exception as e:  # noqa: BLE001
                row["unit_error"] = f"{type(e).__name__}: {e}"
                log.warning("%s %s %s: %s", acc, u.unit, u.domain_id, e)
            rows.append(row)
        status["n_units"] = len(rows)
        keep_default = next((u.mask for u in units if u.unit == "CHAIN_TRIMMED" and u.mask_variant == "default"),
                            np.zeros(s.n_res, bool))
        dom_of = np.array([""] * s.n_res, dtype=object)
        for u in units:
            if u.unit == "DOMAIN" and u.mask_variant == "default":
                dom_of[u.mask] = u.domain_id
        residues = pd.DataFrame({
            "accession": acc, "resnum": s.resnum, "resname": s.res_names, "plddt": s.plddt, "dssp": ss,
            "in_chain_trimmed": keep_default, "domain_id": dom_of.astype(str),
            "sasa": res_sasa if res_sasa is not None else np.nan,
            "depth_mean": res_depth[0] if res_depth is not None else np.nan,
            "depth_max": res_depth[1] if res_depth is not None else np.nan,
        })
    except Exception as e:  # noqa: BLE001
        status.update(status="error", error=f"{type(e).__name__}: {e}")
        log.debug(traceback.format_exc())
    status["seconds"] = round(time.time() - t0, 2)
    return {"rows": rows, "residues": residues, "status": status}


def load_done(out_dir: Path, model_versions: dict[str, int]) -> set[str]:
    parts = sorted((Path(out_dir) / "parts").glob("status-*.parquet"))
    if not parts:
        return set()
    st = pd.concat([pd.read_parquet(p, columns=["accession", "model_version", "metrics_version", "status"])
                    for p in parts])
    st = st[(st["metrics_version"] == METRICS_VERSION) & (st["status"] == "ok")]
    return {a for a, v in zip(st["accession"], st["model_version"]) if model_versions.get(a) == v}


def run_compute(jobs: list[dict], out_dir: Path, workers: int, tag: str, flush_every: int = 200) -> dict:
    parts = Path(out_dir) / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    buf_rows, buf_res, buf_status, n_part, n_done, n_err = [], [], [], 0, 0, 0
    t0 = time.time()

    def flush():
        nonlocal buf_rows, buf_res, buf_status, n_part
        if not buf_status:
            return
        name = f"{tag}-{os.getpid()}-{int(time.time())}-{n_part:04d}"
        if buf_rows:
            pd.DataFrame(buf_rows).to_parquet(parts / f"metrics-{name}.parquet", index=False)
        if buf_res:
            pd.concat(buf_res, ignore_index=True).to_parquet(parts / f"residues-{name}.parquet", index=False)
        pd.DataFrame(buf_status).to_parquet(parts / f"status-{name}.parquet", index=False)   # last: marks done
        buf_rows, buf_res, buf_status = [], [], []
        n_part += 1

    ctx = get_context("fork") if hasattr(os, "fork") else get_context()
    with ctx.Pool(workers, maxtasksperchild=50) as pool:
        for res in pool.imap_unordered(process_protein, jobs, chunksize=1):
            buf_rows.extend(res["rows"])
            if res["residues"] is not None:
                buf_res.append(res["residues"])
            buf_status.append(res["status"])
            n_done += 1
            n_err += res["status"]["status"] != "ok"
            if len(buf_status) >= flush_every:
                flush()
            if n_done % 25 == 0 or n_done == len(jobs):
                rate = n_done / (time.time() - t0)
                log.info("%d/%d proteins (%d errors), %.2f proteins/s", n_done, len(jobs), n_err, rate)
    flush()
    return {"done": n_done, "errors": n_err, "seconds": round(time.time() - t0, 1)}


def collect(out_dir: Path) -> dict[str, pd.DataFrame]:
    parts = Path(out_dir) / "parts"
    out = {}
    for kind in ("metrics", "residues", "status"):
        files = sorted(parts.glob(f"{kind}-*.parquet"))
        out[kind] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    if len(out["status"]):
        # newest status per accession (parts are named by time, so the last one wins)
        out["status"] = out["status"].drop_duplicates(["accession", "metrics_version"], keep="last")
    return out
