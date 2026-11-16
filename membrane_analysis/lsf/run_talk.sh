#!/bin/bash
# Execute the talk-figure notebook on data_<name>/ (run from membrane_analysis/).
# Writes figures_talk_<name>/ (PNG + PDF + tables/) and executed/talk_figures_<name>.executed.ipynb (+ HTML).
# AFTER=1 queues it behind mvsa_protcorr_<name>. N_BOTTOM sets the size of the worst-protein lists (100).
# Usage: bash lsf/run_talk.sh NAME
set -euo pipefail
NAME=${1:?dataset name, e.g. human or yeast}
[ -n "${AFTER:-}" ] || [ -f "data_${NAME}/meta.json" ] || { echo "data_${NAME}/ is not ready: run lsf/prepare.sh first"; exit 1; }
mkdir -p lsf/logs executed
OUT="talk_figures_${NAME}.executed.ipynb"
bsub -q "${CPU_QUEUE:-normal}" -J "mvsa_talk_${NAME}" ${AFTER:+-w "done(mvsa_protcorr_${NAME})"} -n 4 -M 32000 \
     -R "select[mem>32000] rusage[mem=32000]" -o "lsf/logs/talk_${NAME}.log" \
     "source ../.venv/bin/activate && export MVSA_DATA=../data_${NAME} MVSA_FIGURES=../figures_talk_${NAME} MVSA_NBOTTOM=${N_BOTTOM:-100} && jupyter nbconvert --to notebook --execute notebooks/talk_figures.ipynb --output-dir executed --output ${OUT} --ExecutePreprocessor.timeout=7200 && jupyter nbconvert --to html executed/${OUT}"
echo "submitted: figures_talk_${NAME}/ and executed/${OUT} (log lsf/logs/talk_${NAME}.log)"
