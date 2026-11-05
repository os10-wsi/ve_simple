# ve_simple

Given a list of UniProt accessions, `ve_simple` scores every possible single amino-acid
substitution with a panel of variant-effect predictors, attaches ClinVar classifications
(human proteins), stores everything in a SQLite database and flat tables, and benchmarks each
predictor per protein by ROC-AUC (pathogenic vs benign) with standard errors, confidence
intervals and a multi-panel PDF of ROC curves.

## Predictors

| column | source | needs | orientation used for AUC |
|---|---|---|---|
| `alphamissense` | AlphaMissense pathogenicity (per-protein table from AlphaFold DB) | human | higher = pathogenic |
| `esm1v` | ESM-1v masked-marginal log-odds, mean of up to 5 ensemble models | sequence | lower = pathogenic |
| `saprot` | SaProt (650M, AF2) structure-aware masked-marginal log-odds | AlphaFold model | lower = pathogenic |
| `thermompnn` | ThermoMPNN ddG (kcal/mol) | AlphaFold model | higher = destabilising |
| `foldx` | FoldX BuildModel ddG (kcal/mol) | AlphaFold model + FoldX licence | higher = destabilising |
| `blosum62` | BLOSUM62 substitution score | sequence | lower = pathogenic |
| `delta_volume`, `delta_charge`, `delta_hydrophobicity` | mutant minus wild-type side-chain volume (Zamyatnin), nominal charge, Kyte-Doolittle hydropathy | sequence | signed; `abs_*` versions also evaluated |
| `solvent_exposure` | relative solvent accessibility (Shrake-Rupley SASA / max ASA) of the wild-type residue in the AlphaFold model; `plddt` also reported | AlphaFold model | buried = pathogenic |

ThermoMPNN predictions come from two sources, in order:

1. the human proteome-wide ThermoMPNN run released with Liao & Lehner (2026),
   [Zenodo record 18381534](https://zenodo.org/records/18381534) (`thermompnn_outs.tar.gz`, 3.3 GB).
   The archive is downloaded automatically into `data/zenodo/` (or run `ve-simple fetch-zenodo`),
   indexed once, and the files for the requested proteins are extracted in a single streaming pass;
2. otherwise (non-human proteins or proteins missing from the archive) ThermoMPNN is run on the
   AlphaFold model with the official code (cloned automatically into `external/ThermoMPNN`).

The `proteins.tsv` column `thermompnn_source` records which one was used (`zenodo:18381534`, `computed` or `missing`).

## Data sources

* UniProt REST (sequence, organism, gene).
* AlphaFold DB API (model PDB, pLDDT, AlphaMissense per-protein table).
* EBI Proteins API `variation` endpoint for ClinVar classifications mapped onto protein
  coordinates. Only classifications whose source is ClinVar are used. Classes are resolved per
  variant: Pathogenic / Likely pathogenic -> `pathogenic`, Benign / Likely benign -> `benign`,
  VUS -> `vus`; variants with both pathogenic and benign assertions are `conflicting` and excluded
  from AUCs. Use `--strict-labels` to exclude the "likely" categories.
* Hugging Face hub for ESM-1v (`facebook/esm1v_t33_650M_UR90S_1..5`) and SaProt
  (`westlake-repl/SaProt_650M_AF2`) weights.

## Installation

```bash
pip install -e .
```

Optional:

* **FoldX** is licensed software (https://foldxsuite.crg.eu). Point the pipeline at the binary with
  `--foldx-binary /path/to/foldx` or `export FOLDX_BINARY=...`. Without it the `foldx` column is empty.
* **Foldseek** (3Di tokens for SaProt). The pure-Python `mini3di` port is used by default; if a
  `foldseek` binary is on PATH (or `VE_SIMPLE_FOLDSEEK` is set) it is used instead.
* Model weights (about 16 GB for the five ESM-1v models and SaProt) are stored under
  `$HF_HOME`, which the pipeline sets to `<cache>/huggingface` unless you export it yourself.
  Put it on a large filesystem, not a quota-limited home directory.
* A GPU is strongly recommended for ESM-1v and SaProt (650M parameters each; every position of
  every protein needs one masked forward pass). On a 4-core CPU with `--batch-size 8` each model
  takes roughly 2.5 s per residue (about 6 min per model for a 150-residue protein), so for CPU
  runs use `--esm1v-models 1` or `--skip esm1v,saprot`. Sequences longer than 1022 residues are
  scored in a sliding 1022-residue window centred on each position.

### GPU builds of torch

The default PyPI `torch` wheel targets the newest CUDA toolkit; if the cluster's NVIDIA driver is
older you will see `The NVIDIA driver on your system is too old` and the pipeline falls back to
CPU. Install a wheel matching the driver (check `nvidia-smi` for the supported CUDA version), e.g.
for a CUDA 12.4 driver:

```bash
pip uninstall -y torch && pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
```

## Usage

```bash
ve-simple run --ids examples/ids_example.txt --out results/
```

Useful options: `--device cuda`, `--esm1v-models 1`, `--skip foldx,saprot`, `--foldx-scope clinvar`
(FoldX only at positions with ClinVar variants), `--bootstrap 2000`, `--min-per-class 5`,
`--strict-labels`, `--no-thermompnn-compute`, `--force`.

Re-compute AUC statistics and plots from an existing results directory (e.g. with different
bootstrap settings) without re-scoring:

```bash
ve-simple evaluate --out results/ --bootstrap 2000 --min-per-class 5
```

Everything fetched or computed is cached per protein under `cache/<accession>/`, so re-runs only
compute what is missing.

Everything fetched or computed is cached per protein, and `--skip` leaves cached scores in place, so a run can be split into phases and jobs; see `lsf/README.md` for an LSF layout (GPU jobs for the language models, a job array for FoldX, one assembly job).

## Outputs (`results/`)

| file | content |
|---|---|
| `variant_scores.tsv` | one row per protein x position x substitution with every predictor column, `plddt`, and the ClinVar class/label/review status/RCV ids where available |
| `clinvar_variants.tsv` | the ClinVar missense variants collected per human protein (pathogenic, VUS, benign, conflicting) |
| `proteins.tsv` | per-protein metadata: organism, length, AlphaFold model used, ThermoMPNN source, ClinVar counts (pathogenic / VUS / benign / conflicting), number of variants scored per predictor, errors |
| `auc_long.tsv` | one row per human protein x predictor: `n_pathogenic`, `n_benign`, `auc`, `auc_se` (DeLong), `ci_low`/`ci_high` (stratified bootstrap, 1000 resamples by default), `p_value` (AUC != 0.5) |
| `auc_per_protein.tsv` | the same statistics in wide form: one row per human protein, `<predictor>_auc`, `<predictor>_auc_se`, `<predictor>_ci_low`, `<predictor>_ci_high`, `<predictor>_n_scored` for every predictor |
| `roc_curves.pdf` | one ROC panel per human protein (9 per page), one curve per predictor, AUC in the legend |
| `ve_simple.sqlite` | tables `proteins`, `variants`, `clinvar`, `auc` with the same content |
| `summary.json` | counts and output paths |

All AUCs are computed on scores oriented so that higher means more pathogenic (see the table
above), so every value above 0.5 means better-than-random discrimination. An AUC is reported
whenever a protein has at least `--min-per-class` pathogenic and benign variants with a score for
that predictor; otherwise the fields are empty.

## Notes and limitations

* The ClinVar annotations come from the EBI Proteins API rather than directly from NCBI, so a
  variant appears only once UniProt/EBI have mapped it onto the protein; releases lag ClinVar by
  a few weeks.
* ClinVar variants whose reported wild-type residue does not match the UniProt canonical sequence
  (isoform-specific annotations) cannot be scored and are counted in `proteins.tsv` but excluded
  from AUCs.
* The de novo ThermoMPNN route runs the upstream `custom_inference.py` in a subprocess on the
  AlphaFold model (chain A); it needs `torch`, `pytorch-lightning` and `omegaconf` (installed by
  this package) and the checkpoint shipped with the ThermoMPNN repository. Human proteins are
  almost always covered by the Zenodo archive, so this route is mainly used for non-human proteins.
* FoldX runs BuildModel for every substitution; at roughly one to several seconds per mutant a
  400-residue protein takes hours. `--foldx-scope clinvar` limits FoldX to positions that carry
  ClinVar variants, which is all that the AUC benchmark needs.
* AUCs use all ClinVar missense variants with a pathogenic/benign label, irrespective of review
  status; the review status and RCV accessions are kept in `variant_scores.tsv` for filtering.

## Tests

```bash
python -m pytest tests
```
