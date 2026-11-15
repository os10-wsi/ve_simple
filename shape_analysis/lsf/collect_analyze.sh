#!/bin/bash
# Merge the computed parts, add class_final, and run the stage-5 analysis, as one job.
# Can be started from any directory: the job moves to shape_analysis/ itself.
# Usage: bash shape_analysis/lsf/collect_analyze.sh      (or bash lsf/collect_analyze.sh from shape_analysis/)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)        # shape_analysis/
mkdir -p "$HERE/lsf/logs"
bsub -q "${CPU_QUEUE:-normal}" -J protshape_collect -n 4 -M 32000 -R "select[mem>32000] rusage[mem=32000]" \
     -o "$HERE/lsf/logs/analyze.log" \
     "cd $HERE && source ../.venv/bin/activate && protshape collect && protshape classify && protshape analyze"
echo "submitted; log: $HERE/lsf/logs/analyze.log"
