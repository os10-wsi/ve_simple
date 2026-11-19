#!/bin/bash
# Per-protein correlations between predictors and residue context (one pass over the variant table;
# ~20-30 min for human). Needs data_<name>/ from prepare.sh. Run from membrane_analysis/.
# AFTER=1 queues it behind mvsa_prepare_<name>.
# Usage: bash lsf/protcorr.sh NAME [VE_RESULTS]       e.g. bash lsf/protcorr.sh human
set -euo pipefail
NAME=${1:?dataset name, e.g. human or yeast}
VE=${2:-../results_${NAME}}
[ -n "${AFTER:-}" ] || [ -f "data_${NAME}/residues.parquet" ] || { echo "data_${NAME}/ is not ready: run lsf/prepare.sh first"; exit 1; }
mkdir -p lsf/logs
bsub -q "${CPU_QUEUE:-normal}" -J "mvsa_protcorr_${NAME}" ${AFTER:+-w "done(mvsa_prepare_${NAME})"} -n 1 -M 32000 -R "select[mem>32000] rusage[mem=32000]" \
     -o "lsf/logs/protcorr_${NAME}.log" \
     "source ../.venv/bin/activate && python -m mvsa.protcorr --ve $VE --data data_${NAME}"
echo "submitted: data_${NAME}/protein_correlations.parquet (log lsf/logs/protcorr_${NAME}.log)"
