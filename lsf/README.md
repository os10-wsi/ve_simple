# Running on an LSF cluster

The pipeline caches every per-protein intermediate under `cache/<accession>/`, so the work can be
split into jobs by phase and reassembled at the end. Every job must be started from the repository
directory (or pass the same `--cache`, `--data-dir` and `--external-dir`) so that they share the cache.

1. **Prepare shared data once** (login node or a short job; needs ~3.3 GB for the archive):

       ve-simple fetch-zenodo

   The archive is read directly by the jobs; do not unpack it (20,000 files).

2. **Phase 1, GPU jobs** (`lsf/phase1_gpu.sh`): chunks of ~100 proteins; fetches everything,
   computes all predictors except FoldX and writes them to the cache (`--cache-only`: each chunk
   leaves only `proteins.tsv` and `summary.json`). Loading the six language models costs about a
   minute per job, so do not use one job per protein here.

3. **Phase 2, FoldX job array** (`lsf/phase2_foldx.sh`): each element runs FoldX on `PER_JOB`
   proteins (default 10) with several cores, long queue, writing only to the cache. FoldX is the slow part: BuildModel takes a few seconds per mutant, and a
   400-residue protein has 7,600 mutants.

4. **Phase 3, assembly** (`lsf/phase3_assemble.sh`): one short job that reads the cache, builds the
   combined tables, database, AUC statistics and ROC PDF.

Chunk files, job names, logs and `results_phase*` directories are namespaced by the ID file's basename
(e.g. `uniprot_reviewed_9606`), so different proteomes can be submitted independently. Chunks that
failed can be resubmitted with `bash lsf/phase1_resubmit_failed.sh all_ids.txt`. If jobs were ever
killed or duplicated, run `ve-simple check-cache --delete` before resubmitting: it removes cache
files that are unreadable or incomplete so they are recomputed.

Edit the queue names, project, and module/venv lines at the top of each script for your cluster.

## Proteome-scale runs (e.g. all reviewed human and yeast proteins)

Run each proteome as its own set of jobs with its own output directory; the cache is shared.
The ID files can be UniProt TSV downloads (the first accession on each line is used).

    ve-simple fetch-zenodo                                     # once
    bash lsf/phase1_gpu.sh uniprot_reviewed_9606.txt 100       # ~205 GPU jobs
    bash lsf/phase1_gpu.sh uniprot_reviewed_559292.txt 100     # ~68 GPU jobs (ThermoMPNN computed de novo)
    FOLDX_BINARY=... bash lsf/phase2_foldx.sh uniprot_reviewed_9606.txt 8
    FOLDX_BINARY=... bash lsf/phase2_foldx.sh uniprot_reviewed_559292.txt 8
    bash lsf/phase3_assemble.sh uniprot_reviewed_9606.txt results_human
    bash lsf/phase3_assemble.sh uniprot_reviewed_559292.txt results_yeast

Budget: a 100-protein chunk took 79 minutes on one GPU (~50 s per protein for all six language
models), so the human proteome is ~270 GPU-hours spread over the array. FoldX saturation mutagenesis is far larger:
a few seconds per mutant times 19 x length, i.e. several CPU-hours for an average protein and
weeks for the largest ones, so ~100,000 core-hours per proteome. `FOLDX_SCOPE=clinvar` restricts
FoldX to positions with ClinVar variants (all the AUC benchmark needs) and cuts this ~50-fold;
it is only meaningful for human proteins. Assembly streams proteins in batches, so it needs little
memory, but the human `variant_scores.tsv` will be ~190 million rows (tens of GB); the SQLite
database next to it has the same content indexed by accession.

## File-count budget

Lustre quotas usually limit the number of files as well as the space. After a protein is
finished its cache directory holds about ten files (UniProt and AlphaFold metadata, the model,
the AlphaMissense table, the ClinVar table and one table per predictor), i.e. ~200,000 files for
the human proteome and ~60,000 for yeast. Temporary files (per-model ESM-1v matrices, FoldX work
directories, de novo ThermoMPNN outputs) are deleted as soon as the protein's table is written.
Each LSF job writes one log file.

Caches from versions before this change can be shrunk in place:

    ve-simple compact-cache --remove-zenodo-extracted

Check usage with `lfs quota -h -u $USER /lustre/scratch125` (and `-g <group>` for group quotas).
