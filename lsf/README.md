# Running on an LSF cluster

The pipeline caches every per-protein intermediate under `cache/<accession>/`, so the work can be
split into jobs by phase and reassembled at the end. Every job must be started from the repository
directory (or pass the same `--cache`, `--data-dir` and `--external-dir`) so that they share the cache.

1. **Prepare shared data once** (login node or a short job; needs ~3.3 GB for the archive):

       ve-simple fetch-zenodo --extract-all

2. **Phase 1, GPU jobs** (`lsf/phase1_gpu.sh`): chunks of ~100 proteins; fetches everything,
   computes all predictors except FoldX. Loading the six language models costs about a minute per
   job, so do not use one job per protein here.

3. **Phase 2, FoldX job array** (`lsf/phase2_foldx.sh`): one array element per protein, several
   cores each, long queue. FoldX is the slow part: BuildModel takes a few seconds per mutant, and a
   400-residue protein has 7,600 mutants.

4. **Phase 3, assembly** (`lsf/phase3_assemble.sh`): one short job that reads the cache, builds the
   combined tables, database, AUC statistics and ROC PDF.

Edit the queue names, project, and module/venv lines at the top of each script for your cluster.

## Proteome-scale runs (e.g. all reviewed human and yeast proteins)

Run each proteome as its own set of jobs with its own output directory; the cache is shared.
The ID files can be UniProt TSV downloads (the first accession on each line is used).

    ve-simple fetch-zenodo --extract-all                       # once
    bash lsf/phase1_gpu.sh uniprot_reviewed_9606.txt 100       # ~205 GPU jobs
    bash lsf/phase1_gpu.sh uniprot_reviewed_559292.txt 100     # ~68 GPU jobs (ThermoMPNN computed de novo)
    FOLDX_BINARY=... bash lsf/phase2_foldx.sh uniprot_reviewed_9606.txt 8
    FOLDX_BINARY=... bash lsf/phase2_foldx.sh uniprot_reviewed_559292.txt 8
    bash lsf/phase3_assemble.sh uniprot_reviewed_9606.txt results_human
    bash lsf/phase3_assemble.sh uniprot_reviewed_559292.txt results_yeast

Budget: on one GPU the six language models take roughly 1-3 minutes per protein, so the human
proteome is ~500-1000 GPU-hours spread over the array. FoldX saturation mutagenesis is far larger:
a few seconds per mutant times 19 x length, i.e. several CPU-hours for an average protein and
weeks for the largest ones, so ~100,000 core-hours per proteome. `FOLDX_SCOPE=clinvar` restricts
FoldX to positions with ClinVar variants (all the AUC benchmark needs) and cuts this ~50-fold;
it is only meaningful for human proteins. Assembly streams proteins in batches, so it needs little
memory, but the human `variant_scores.tsv` will be ~190 million rows (tens of GB); the SQLite
database next to it has the same content indexed by accession.
