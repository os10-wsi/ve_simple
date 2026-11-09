#!/bin/bash
# Phase 2: FoldX, as a job array; each element runs PER_JOB proteins one after another.
# Usage:  FOLDX_BINARY=/path/to/foldx [FOLDX_SCOPE=all|clinvar] [PER_JOB=10] \
#           bash lsf/phase2_foldx.sh all_ids.txt [cores_per_job]
# Results go only to cache/<accession>/scores_foldx.tsv; FoldX work files are deleted as it goes.
# A rerun of the same command skips proteins that are already finished.
set -euo pipefail
IDS=${1:?id list}
CORES=${2:-8}
QUEUE=${CPU_QUEUE:-long}
PER_JOB=${PER_JOB:-10}
: "${FOLDX_BINARY:?set FOLDX_BINARY to the FoldX executable}"
TAG=$(basename "$IDS"); TAG=${TAG%.*}
mkdir -p lsf/logs lsf/chunks
LIST="lsf/chunks/foldx_ids_${TAG}.txt"
# only proteins without a finished FoldX table
python -c '
import sys
from pathlib import Path
from ve_simple.utils import read_id_list
for a in read_id_list(sys.argv[1]):
    if not (Path("cache") / a / "scores_foldx.tsv").exists():
        print(a)' "$IDS" > "$LIST"
N=$(wc -l < "$LIST")
[ "$N" -gt 0 ] || { echo "nothing to do for $TAG"; exit 0; }
NJOBS=$(( (N + PER_JOB - 1) / PER_JOB ))
echo "$N proteins without FoldX in $NJOBS array elements (tag: $TAG)"
bsub -q "$QUEUE" -J "ve_foldx_${TAG}[1-${NJOBS}]" -n "$CORES" -M 8000 -R "select[mem>8000] rusage[mem=8000] span[hosts=1]" \
     -o "lsf/logs/foldx_${TAG}_%I.log" <<EOS
source .venv/bin/activate
export FOLDX_BINARY=$FOLDX_BINARY
FIRST=\$(( (LSB_JOBINDEX - 1) * $PER_JOB + 1 ))
LAST=\$(( LSB_JOBINDEX * $PER_JOB ))
ve-simple foldx --id \$(sed -n "\${FIRST},\${LAST}p" "$LIST") \
          --foldx-workers $CORES --foldx-scope ${FOLDX_SCOPE:-all}
EOS
