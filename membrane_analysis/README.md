# Membrane vs soluble: variant-effect prediction and structure

Notebooks that put every variant-effect predictor (ve_simple) and every structural measurement
(shape_analysis) side by side for **multipass membrane proteins vs every other protein class**, to
test whether membrane proteins are a blind spot for current prediction: (i) worse clinical
performance, (ii) compressed or mis-calibrated scores and more predictor disagreement, (iii) distinct
structural properties.

## Inputs

* `results_human/` from the ve_simple assembly (`lsf/phase3_assemble.sh`): `variant_scores.tsv`,
  `proteins.tsv`, `auc_long.tsv`.
* `shape_analysis/results/` after `protshape collect`: `classes.parquet`, `annotations.parquet`,
  `metrics.parquet`, `residues.parquet`, `status.parquet`.

## 1. Prepare the compact tables (once, on the cluster)

```bash
pip install -e membrane_analysis
cd membrane_analysis && bash lsf/prepare.sh ../results_human ../shape_analysis/results
```

This streams the full variant table once and writes `data/` (a few hundred MB): every
ClinVar-labelled variant with all scores and residue context, a 1% random sample of all possible
substitutions, per-protein score summaries, per-protein AUCs, shape metrics and per-residue context.

## 2. Notebooks

| notebook | purpose |
|---|---|
| `notebooks/membrane_vs_soluble.ipynb` | the narrative: sections (i) performance, (ii) dynamic range and error, (iii) structure, each ending in a generated takeaway; figures saved to `figures/` as PDF and PNG; summary table `figures/summary_multipass_vs_soluble.tsv` |
| `notebooks/explore.ipynb` | browse any predictor, any structural metric, or one protein, for every class |

Run the narrative on the cluster (it bootstraps over thousands of proteins; ~30-60 minutes on the full data):

```bash
bash lsf/run_notebook.sh
```

It writes `executed/membrane_vs_soluble.executed.ipynb`, an HTML copy next to it, and `figures/`.
Open the explorer interactively in Jupyter on a compute node.

Parameters at the top of each notebook: `INCLUDE_LIKELY`, `MIN_STARS` (ClinVar review filter),
`N_BOOT`, `MIN_PER_CLASS`; environment variables `MVSA_DATA`, `MVSA_FIGURES`, `MVSA_NBOOT`.
Regenerate the notebooks after editing `scripts/make_notebooks.py`.

## Definitions

* **Classes** come from UniProt annotation (`class_annotation`): MULTIPASS (>= 2 helical TM segments),
  SINGLE_PASS, BETA_BARREL_MEMBRANE, SOLUBLE, AMBIGUOUS.
* **Orientation**: every predictor is multiplied by its sign so that higher = more likely pathogenic;
  AUC > 0.5 is always better than chance.
* **Labels**: pathogenic + likely pathogenic vs benign + likely benign (VUS and conflicting kept separately).
* **Confidence intervals** resample whole proteins (variants of one protein are not independent);
  p-values are BH-adjusted across predictors.
* **Location** of a residue in a membrane protein: TM lipid-facing (in a UniProt TM segment and exposed in
  the AlphaFold model, relative SASA >= 0.2), TM buried, juxtamembrane (within 5 residues), extramembrane
  exposed / buried; soluble proteins: exposed / buried. Residue context is withheld where the AlphaFold model
  sequence differs from the current UniProt sequence.
* **Matched AUC**: within strata of residue pLDDT x burial x the protein's pathogenic fraction, weighted to
  the multipass strata. **Within-protein AUC**: AUC inside each protein, averaged with n_P x n_B weights.
* **Thresholds** without a published cut-off are tuned on soluble proteins (Youden's J) and applied unchanged
  to every class; calibration maps are fitted on soluble proteins.

## Tests

```bash
python -m pytest membrane_analysis/tests
```
