# Running more datasets: model proteomes and membrane superfamilies

Every dataset goes through the same four stages as the human and yeast proteomes:

1. **ve_simple phase 1** (GPU): fetch UniProt entries, AlphaFold models and every predictor into the shared cache.
2. **ve_simple phase 3** (CPU array + merge): one `results_<name>/` per dataset.
3. **protshape** (CPU): shape metrics for every protein in the cache (shared by all datasets; finished proteins are skipped).
4. **mvsa** (CPU): `data_<name>/` and the notebooks.

Datasets are identified by an ID file. Its basename tags the phase-1/3 jobs and chunk folders, so datasets never collide.

## Budget before you start

| | proteins | GPU hours (phase 1)* | cache files** |
|---|---:|---:|---:|
| 20 families × 500 representatives | ≤ 10,000 | ~150 | ~110,000 |
| fly (7227) | 3,912 | ~55 | ~43,000 |
| worm (6239) | 4,490 | ~60 | ~49,000 |
| E. coli K-12 (83333) | 4,531 | ~60 | ~50,000 |
| fission yeast (284812) | 5,129 | ~70 | ~56,000 |
| zebrafish (7955) | 3,388 | ~45 | ~37,000 |
| Dictyostelium (44689) | 4,163 | ~55 | ~46,000 |
| rat (10116) | 8,239 | ~110 | ~91,000 |
| Arabidopsis (3702) | 16,421 | ~220 | ~180,000 |
| mouse (10090) | 17,285 | ~230 | ~190,000 |

\* About 80 minutes per 100 proteins, as measured on human. Only human ThermoMPNN comes from the Zenodo archive; every other protein has
ThermoMPNN computed on the GPU, as for yeast.
\*\* About 11 files per protein: the cache plus TED domains and shape parts.

**File quota.** The human and yeast runs already use roughly 300,000 files of the 700,000 allowed. The families plus fly, worm and
E. coli fit (≈ 250,000 more). Arabidopsis or mouse on top would not. Check first and compact if needed:

    lfs quota -h -u $USER /lustre/scratch125
    ve-simple compact-cache --remove-zenodo-extracted

## A. Model proteomes

From the repository root, with the venv active. `P` is where the ID lists live.

    P=/lustre/scratch125/gengen/projects_v2/mp_stability/retrieve_proteomes
    bash scripts/fetch_proteome_ids.sh $P 7227 6239 83333           # fly, worm, E. coli (add more taxids as budget allows)

    bash lsf/phase1_gpu.sh $P/uniprot_reviewed_7227.txt 100         # one GPU array per proteome
    bash lsf/phase1_gpu.sh $P/uniprot_reviewed_6239.txt 100
    bash lsf/phase1_gpu.sh $P/uniprot_reviewed_83333.txt 100

When the GPU jobs have finished, check them and resubmit failures exactly as for human and yeast (`ve-simple incomplete-chunks`,
`lsf/phase1_resubmit_failed.sh`). Then assemble:

    bash lsf/phase3_assemble.sh $P/uniprot_reviewed_7227.txt results_fly
    bash lsf/phase3_assemble.sh $P/uniprot_reviewed_6239.txt results_worm
    bash lsf/phase3_assemble.sh $P/uniprot_reviewed_83333.txt results_ecoli

## B. Membrane superfamilies

`sets/families/families.tsv` lists the 20 largest integral-membrane Pfam families, one per Pfam clan (superfamily), from InterPro
counts in November 2026. `sets/families/candidates.tsv` holds all 70 candidates checked. To use different families, edit
`families.tsv`; for example, swap in Sugar_tr (PF00083) or the olfactory receptors (7tm_4, PF13853), which share a clan with MFS_1
and 7tm_1.

For each family, `scripts/family_sets.py` downloads the UniProtKB members that are not fragments, have an AlphaFold model and are
100–1,500 residues long. It removes sequences with non-standard residues and clusters the rest with MMseqs2 (30% identity, 80%
coverage). If that gives fewer than 500 clusters, it re-clusters at 50%, 70% and then 90%. It keeps the representatives of the 500
largest clusters (`--pick random` takes 500 clusters at random instead) and records each one's organism and domain of life.

MMseqs2 is not part of the venv. Install it once, either with `conda install -c conda-forge -c bioconda mmseqs2` or as the static
binary from https://mmseqs.com/latest/ (`mmseqs-linux-avx2.tar.gz`). Then:

    MMSEQS=/path/to/mmseqs bash lsf/family_sets.sh 500 16          # one CPU job, several hours for MFS (~700k sequences)
    tail -f lsf/logs/family_sets.log

This writes `sets/families/ids/<PFAM>_<short>.txt`, `all_family_ids.txt` (the union) and `family_membership.tsv`. The downloads in
`sets/families/work/` take several GB; delete them once the lists look right. Then run the families as one dataset:

    bash lsf/phase1_gpu.sh sets/families/all_family_ids.txt 100
    bash lsf/phase3_assemble.sh sets/families/all_family_ids.txt results_families

## C. Shape metrics (once, after all phase-1 jobs of the new datasets)

    cd shape_analysis
    protshape index-cache --cache ../cache                         # picks up every cached protein
    protshape annotate --threads 4 && protshape annotate --threads 2   # second pass retries failed TED queries
    protshape classify
    bash lsf/compute_array.sh 50 8                                  # only new proteins are computed
    SKIP_ANALYZE=1 bash lsf/collect_analyze.sh                      # collect + classify; keeps the human/yeast findings.md
    cd ..

## D. Membrane analysis

From `membrane_analysis/`, for each dataset (names follow `results_<name>`):

    bash lsf/prepare.sh ../results_fly                    # -> data_fly/
    bash lsf/protcorr.sh fly
    AFTER=1 bash lsf/run_talk.sh fly
    AFTER=1 bash lsf/run_notebook.sh fly                  # the full narrative notebook (clinical sections skip themselves)

    bash lsf/prepare.sh ../results_families               # -> data_families/
    bash lsf/protcorr.sh families
    AFTER=1 bash lsf/run_families.sh                      # cross-family notebook -> figures_families/

To put several proteomes on the talk structure figure:

    STRUCT_SETS=human,yeast,fly,worm,ecoli bash lsf/run_talk.sh human

## What the families notebook shows

* **F1:** the families, their size and their domain-of-life mix.
* **F2–F3:** per-family agreement between predictors (median per-protein Spearman ρ over all substitutions), with ESM-1v vs ThermoMPNN as the headline pair.
* **F4–F5:** how widely each predictor spreads its scores, and how strongly it penalises TM substitutions, per family.
* **F6:** each family's structural fingerprint.
* **F7:** which protein properties go with disagreement, both pooled over all proteins and within families.
* **Table:** the least-agreeing proteins in each family.

Tables are in `figures_families/tables/`.
