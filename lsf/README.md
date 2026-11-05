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
