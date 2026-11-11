#!/bin/bash
# Full proteome run of `protshape compute` as an LSF job array (run from shape_analysis/).
# Each element computes every NSHARDS-th protein with WORKERS processes and writes its own parts,
# so elements never touch the same files. Rerunning the same command skips finished proteins.
# Usage:  bash lsf/compute_array.sh [NSHARDS] [WORKERS]
set -euo pipefail
NSHARDS=${1:-50}
WORKERS=${2:-8}
QUEUE=${CPU_QUEUE:-normal}
MEM=${HOST_MEM:-16000}
mkdir -p lsf/logs
bsub -q "$QUEUE" -J "protshape[1-${NSHARDS}]" -n "$WORKERS" -M "$MEM" \
     -R "select[mem>$MEM] rusage[mem=$MEM] span[hosts=1]" -o "lsf/logs/compute_%I.log" \
     "source ../.venv/bin/activate && protshape compute --shard \$((LSB_JOBINDEX - 1)) --nshards $NSHARDS --workers $WORKERS"
