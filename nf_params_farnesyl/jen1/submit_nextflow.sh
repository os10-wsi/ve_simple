#!/bin/bash
#BSUB -J nextflow
#BSUB -q oversubscribed
#BSUB -n 1
#BSUB -M 4000
#BSUB -R "select[mem>4000] rusage[mem=4000]"
#BSUB -o nextflow.%J.out
#BSUB -e nextflow.%J.err

set -euo pipefail

# Run directory for this analysis
RUN_DIR=/lustre/scratch125/gengen/projects_v2/mp_stability/2026_paper/experimental_datasets/farnesyl_reporter/med_20261711/jen1
cd "$RUN_DIR"

module load HGI/pipelines/deepmutscan/v07_09_2026
module load ISG/conda
conda activate test

# Keep Singularity caches off the home quota (shared between runs)
SING=/lustre/scratch125/gengen/projects_v2/mp_stability/singularity
mkdir -p "$SING/cache" "$SING/tmp" "$SING/images"

export SINGULARITY_CACHEDIR=$SING/cache          # keep: shared download cache
export SINGULARITY_TMPDIR=${TMPDIR:-/tmp}        # change: temporary build space
export APPTAINER_CACHEDIR=$SING/cache            # keep
export APPTAINER_TMPDIR=${TMPDIR:-/tmp}          # change
export NXF_SINGULARITY_CACHEDIR=$SING/images     # keep: finished images
export NXF_VER=26.04.6

nextflow run nf-core/deepmutscan -r dev -profile sanger,singularity \
    -name JEN1_farnesyl_med_nf -resume \
    -work-dir "$RUN_DIR/work" \
    -params-file nf-params.json
