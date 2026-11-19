#!/bin/bash
# Execute the cross-family notebook on data_families/ (run from membrane_analysis/).
# Writes figures_families/ (PNG + PDF + tables/) and executed/families.executed.ipynb (+ HTML).
# AFTER=1 queues it behind mvsa_protcorr_families.
# Usage: bash lsf/run_families.sh [NAME]          (NAME defaults to families: data_families/)
set -euo pipefail
NAME=${1:-families}
[ -n "${AFTER:-}" ] || [ -f "data_${NAME}/meta.json" ] || { echo "data_${NAME}/ is not ready: run lsf/prepare.sh first"; exit 1; }
MEMB=${MEMBERSHIP:-../sets/families/family_membership.tsv}
[ -f "$MEMB" ] || { echo "$MEMB not found: run lsf/family_sets.sh first (or set MEMBERSHIP=...)"; exit 1; }
mkdir -p lsf/logs executed
bsub -q "${CPU_QUEUE:-normal}" -J "mvsa_families_${NAME}" ${AFTER:+-w "done(mvsa_protcorr_${NAME})"} -n 2 -M 32000 \
     -R "select[mem>32000] rusage[mem=32000]" -o "lsf/logs/families_${NAME}.log" \
     "source ../.venv/bin/activate && export MVSA_DATA=../data_${NAME} MVSA_FIGURES=../figures_${NAME} MVSA_MEMBERSHIP=../$MEMB && jupyter nbconvert --to notebook --execute notebooks/families.ipynb --output-dir executed --output ${NAME}.executed.ipynb --ExecutePreprocessor.timeout=7200 && jupyter nbconvert --to html executed/${NAME}.executed.ipynb"
echo "submitted: figures_${NAME}/ and executed/${NAME}.executed.ipynb (log lsf/logs/families_${NAME}.log)"
