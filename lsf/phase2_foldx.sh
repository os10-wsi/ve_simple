#!/bin/bash
# Phase 2: FoldX saturation mutagenesis, one job-array element per protein.
# Usage:  FOLDX_BINARY=/path/to/foldx [FOLDX_SCOPE=all|clinvar] bash lsf/phase2_foldx.sh all_ids.txt [cores_per_job]
set -euo pipefail
IDS=${1:?id list}
CORES=${2:-8}
QUEUE=${CPU_QUEUE:-long}
: "${FOLDX_BINARY:?set FOLDX_BINARY to the FoldX executable}"
TAG=$(basename "$IDS"); TAG=${TAG%.*}
mkdir -p lsf/logs lsf/chunks
python -c 'import sys; from ve_simple.utils import read_id_list; print("\n".join(read_id_list(sys.argv[1])))' "$IDS" > "lsf/chunks/foldx_ids_${TAG}.txt"
N=$(wc -l < "lsf/chunks/foldx_ids_${TAG}.txt")
bsub -q "$QUEUE" -J "ve_foldx_${TAG}[1-${N}]" -n "$CORES" -M 8000 -R "select[mem>8000] rusage[mem=8000] span[hosts=1]" \
     -o "lsf/logs/foldx_${TAG}_%I.out" -e "lsf/logs/foldx_${TAG}_%I.err" <<EOS
source .venv/bin/activate
ACC=\$(sed -n "\${LSB_JOBINDEX}p" "lsf/chunks/foldx_ids_${TAG}.txt")
export FOLDX_BINARY=$FOLDX_BINARY
ve-simple run --id "\$ACC" --out results_phase2/${TAG}/\$ACC --skip esm1v,saprot,thermompnn,alphamissense \
          --foldx-workers $CORES --foldx-scope ${FOLDX_SCOPE:-all} --no-thermompnn-download --no-pdf
EOS
