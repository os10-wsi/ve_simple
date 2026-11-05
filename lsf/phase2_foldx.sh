#!/bin/bash
# Phase 2: FoldX saturation mutagenesis, one job-array element per protein.
# Usage:  FOLDX_BINARY=/path/to/foldx bash lsf/phase2_foldx.sh all_ids.txt [cores_per_job]
set -euo pipefail
IDS=${1:?id list}
CORES=${2:-8}
QUEUE=${CPU_QUEUE:-long}
: "${FOLDX_BINARY:?set FOLDX_BINARY to the FoldX executable}"
mkdir -p lsf/logs
grep -v '^#' "$IDS" | awk 'NF' > lsf/foldx_ids.txt
N=$(wc -l < lsf/foldx_ids.txt)
bsub -q "$QUEUE" -J "ve_foldx[1-${N}]" -n "$CORES" -M 8000 -R "select[mem>8000] rusage[mem=8000] span[hosts=1]" \
     -o "lsf/logs/foldx_%I.out" -e "lsf/logs/foldx_%I.err" <<EOS
source .venv/bin/activate
ACC=\$(sed -n "\${LSB_JOBINDEX}p" lsf/foldx_ids.txt)
export FOLDX_BINARY=$FOLDX_BINARY
ve-simple run --id "\$ACC" --out results_phase2/\$ACC --skip esm1v,saprot,thermompnn,alphamissense \
          --foldx-workers $CORES --no-thermompnn-download
EOS
