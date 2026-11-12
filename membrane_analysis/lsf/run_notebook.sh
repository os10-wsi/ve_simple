#!/bin/bash
# Execute the narrative notebook on the full data as a job (run from membrane_analysis/).
# Writes executed/membrane_vs_soluble.executed.ipynb, an HTML copy, and figures/ (PDF + PNG + summary TSV).
set -euo pipefail
mkdir -p lsf/logs executed
bsub -q "${CPU_QUEUE:-normal}" -J mvsa_notebook -n 4 -M 32000 -R "select[mem>32000] rusage[mem=32000]" \
     -o lsf/logs/notebook.log \
     "source ../.venv/bin/activate && jupyter nbconvert --to notebook --execute notebooks/membrane_vs_soluble.ipynb --output-dir executed --output membrane_vs_soluble.executed.ipynb --ExecutePreprocessor.timeout=7200 && jupyter nbconvert --to html executed/membrane_vs_soluble.executed.ipynb"
