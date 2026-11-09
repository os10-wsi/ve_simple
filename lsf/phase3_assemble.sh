#!/bin/bash
# Phase 3: assemble all cached scores into one results directory with AUC tables and ROC PDF.
# Usage:  bash lsf/phase3_assemble.sh all_ids.txt results_final
set -euo pipefail
IDS=${1:?id list}
OUT=${2:-results_final}
QUEUE=${CPU_QUEUE:-normal}
mkdir -p lsf/logs
TAG=$(basename "$IDS"); TAG=${TAG%.*}
bsub -q "$QUEUE" -J "ve_assemble_${TAG}" -n 2 -M 16000 -R "select[mem>16000] rusage[mem=16000]" \
     -o "lsf/logs/assemble_${TAG}.log" \
     "source .venv/bin/activate && ve-simple run --ids $IDS --out $OUT --skip esm1v,saprot,foldx,thermompnn --no-thermompnn-download --bootstrap 2000"
