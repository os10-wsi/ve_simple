#!/bin/bash
# Build membrane_analysis/data from the assembled ve_simple and shape results (one pass over the
# full variant table; ~1 hour for the human proteome). Run from membrane_analysis/.
# Usage: bash lsf/prepare.sh [VE_RESULTS] [SHAPE_RESULTS]
set -euo pipefail
VE=${1:-../results_human}
SHAPE=${2:-../shape_analysis/results}
mkdir -p lsf/logs
bsub -q "${CPU_QUEUE:-normal}" -J mvsa_prepare -n 2 -M 48000 -R "select[mem>48000] rusage[mem=48000]" \
     -o lsf/logs/prepare.log \
     "source ../.venv/bin/activate && python -m mvsa.prepare --ve $VE --shape $SHAPE --out data"
