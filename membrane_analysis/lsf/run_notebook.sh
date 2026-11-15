#!/bin/bash
# Execute the narrative notebook on data_<name>/ as a job (run from membrane_analysis/ after prepare.sh).
# Writes executed/membrane_vs_soluble_<name>.executed.ipynb (+ HTML) and figures_<name>/.
# Several names can run at the same time.
# Usage: bash lsf/run_notebook.sh NAME        e.g. human, yeast
set -euo pipefail
NAME=${1:?dataset name, e.g. human or yeast}
[ -f "data_${NAME}/meta.json" ] || { echo "data_${NAME}/ is not ready: run lsf/prepare.sh first"; exit 1; }
mkdir -p lsf/logs executed
OUT="membrane_vs_soluble_${NAME}.executed.ipynb"
bsub -q "${CPU_QUEUE:-normal}" -J "mvsa_notebook_${NAME}" -n 4 -M 32000 -R "select[mem>32000] rusage[mem=32000]" \
     -o "lsf/logs/notebook_${NAME}.log" \
     "source ../.venv/bin/activate && export MVSA_DATA=../data_${NAME} MVSA_FIGURES=../figures_${NAME} && jupyter nbconvert --to notebook --execute notebooks/membrane_vs_soluble.ipynb --output-dir executed --output ${OUT} --ExecutePreprocessor.timeout=7200 && jupyter nbconvert --to html executed/${OUT}"
echo "submitted: executed/${OUT} and figures_${NAME}/ (log lsf/logs/notebook_${NAME}.log)"
