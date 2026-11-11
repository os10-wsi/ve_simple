# protshape: proteome-wide protein shape analysis

Shape, size, compactness and hydrophobicity-distribution metrics for every reviewed human and yeast
protein in the ve_simple AlphaFold cache, per protein and per TED domain, and an analysis of which
metrics are size-invariant and which separate membrane from soluble proteins.

Status: **Stage 0 complete, awaiting review before the full run** (see `docs/stage0.md`).

## Install

From the ve_simple repository root, in the existing venv:

```bash
pip install -e shape_analysis
```

## Commands (run from `shape_analysis/`)

| command | what it does | output (under `results/`) |
|---|---|---|
| `protshape index-cache --cache ../cache --ids <idfiles>` | index the ve_simple cache by accession, check coverage | `index.parquet`, `missing_<list>.txt` |
| `protshape annotate` | UniProt annotations from cached `uniprot.json`; TED domains from ted.cathdb.info (resumable) | `annotations.parquet`, `ted/` |
| `protshape classify` | protein classes; after `collect` also the structural cross-check | `classes.parquet` |
| `protshape select-stage0 --n 200` | 50 MULTIPASS + 50 SOLUBLE per organism | `stage0_ids.txt` |
| `protshape compute [--ids ...] [--shard i --nshards N] --workers W` | all units and metrics, resumable, sharded | `compute/parts/` |
| `protshape collect` | merge parts, add class_final | `metrics.parquet`, `residues.parquet`, `status.parquet` |
| `protshape sanity` | stage-4 sanity table | `sanity.tsv` |
| `protshape analyze` | stage-5 analyses and figures | `analysis/`, `findings.md` |

`scripts/stage0.sh` runs Stage 0 end to end on the cluster; `lsf/compute_array.sh` runs the full
compute as an LSF array; `notebooks/analysis.ipynb` is the analysis notebook (regenerate with
`python scripts/make_notebook.py`; set `PROTSHAPE_WORK` to point it at a results directory).

## Data and versions

| source | version / access |
|---|---|
| AlphaFold DB models | ve_simple cache, `cache/<ACC>/AF-<ACC>-F1-model.pdb`; version per protein in `index.parquet` (v6 at download) |
| UniProt annotations | cached `uniprot.json` (full UniProtKB entries); `uniprot_last_update` per entry in `annotations.parquet` |
| TED domains | ted.cathdb.info API v1 (`/uniprot/summary/<acc>`); TED is built on AFDB v4 models, so each domain's `md5_domain` is checked against the cached model sequence (shifts up to +-30 residues; otherwise dropped) |

Only reviewed entries in the two proteome ID files are analysed, F1 models only (fragmented long
proteins are flagged `is_fragment`). PAE is not used by any metric and is not downloaded.

## Units, masks and classes

* Exclusion mask: pLDDT < 70, outside every verified TED domain, or in a confident segment < 10 residues.
* Units: CHAIN_FULL, CHAIN_TRIMMED, DOMAIN (>= 40 residues and >= 3 SSEs after masking), TM_CORE (TM ± 3).
* Mask variants for the sensitivity analysis: `default`, `plddt50`, `plddt90`, `no_ted` (`mask_variant` column).
* `class_annotation` from UniProt only (used for all main analyses); `class_final` after the structural
  belt cross-check (MULTIPASS, beta barrels and SOLUBLE that disagree move to AMBIGUOUS; SINGLE_PASS
  unchanged), used as a sensitivity analysis.
* Domain labels from overlap with UniProt TM/intramembrane features, using `class_annotation`.

## Metrics

Size (Rg mass-weighted and CA, Rmin empirical and sphere, Dmax, Rh proxy, CA contact density);
gyration-tensor descriptors all-atom and CA (`ca_` prefix); convex hull of vdW surfaces, solidity,
sphericity (hull and molecular surface), SASA; 1 A voxel pipeline (vdW and molecular volume and area,
interstitial voids, enclosed cavities and their lining, burial depth per atom/residue); hydrophobicity
distribution on three scales (`_kd`, `_ww`, `_atom`): H(depth) profile, slope and inversion index,
hydrophobic Rg ratio, belt profile on principal axis 1 and the 200-direction slab scan, radial profile,
hydrophobic moment; DSSP fractions and helix statistics. Deviations from the spec and their reasons
are listed in `docs/stage0.md`.

## Tests

```bash
python -m pytest shape_analysis/tests
```

Synthetic point clouds and pseudo-atom assemblies with known answers (sphere, prolate/oblate
ellipsoids, rod, dumbbell, hollow shell cavity, layered hydrophobicity, banded cylinder, flat disc with
an off-axis band, molecular volume/area calibration).

## Runtime

Stage 0: 204 proteins in 5 minutes on 4 cores (5.7 s per protein per core on average). The full
~27,000 proteins need about 45 core-hours, e.g. `bash lsf/compute_array.sh 50 8`.
