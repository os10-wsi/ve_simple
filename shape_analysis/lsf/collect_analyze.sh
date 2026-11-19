#!/bin/bash
# Merge the computed parts, add class_final, and run the stage-5 analysis, as one job.
# Can be started from any directory: the job moves to shape_analysis/ itself.
# Usage: bash shape_analysis/lsf/collect_analyze.sh      (or bash lsf/collect_analyze.sh from shape_analysis/)
# SKIP_ANALYZE=1 runs only collect + classify (what mvsa needs). Use it once datasets other than the human and yeast
# proteomes are in the cache: the stage-5 report (findings.md) is stratified by organism and written for those two.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)        # shape_analysis/
mkdir -p "$HERE/lsf/logs"
bsub -q "${CPU_QUEUE:-normal}" -J protshape_collect -n 4 -M 32000 -R "select[mem>32000] rusage[mem=32000]" \
     -o "$HERE/lsf/logs/analyze.log" \
     "cd $HERE && source ../.venv/bin/activate && protshape collect && protshape classify $( [ "${SKIP_ANALYZE:-0}" = 1 ] || echo "&& protshape analyze" )"
echo "submitted; log: $HERE/lsf/logs/analyze.log"
