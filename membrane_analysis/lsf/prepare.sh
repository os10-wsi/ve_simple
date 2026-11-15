#!/bin/bash
# Build the compact tables from one ve_simple assembly (one pass over its variant table; ~30-60 min
# for human). Run from membrane_analysis/. Several proteomes can run at the same time: each writes to
# data_<name>/ and lsf/logs/prepare_<name>.log, with <name> taken from the results folder
# (results_human -> human) unless given.
# Usage: bash lsf/prepare.sh VE_RESULTS [SHAPE_RESULTS] [NAME]
set -euo pipefail
VE=${1:?ve_simple results directory, e.g. ../results_human}
SHAPE=${2:-../shape_analysis/results}
NAME=${3:-$(basename "$VE" | sed 's/^results_//')}
mkdir -p lsf/logs
bsub -q "${CPU_QUEUE:-normal}" -J "mvsa_prepare_${NAME}" -n 2 -M 48000 -R "select[mem>48000] rusage[mem=48000]" \
     -o "lsf/logs/prepare_${NAME}.log" \
     "source ../.venv/bin/activate && python -m mvsa.prepare --ve $VE --shape $SHAPE --out data_${NAME}"
echo "submitted: data_${NAME}/ (log lsf/logs/prepare_${NAME}.log)"
