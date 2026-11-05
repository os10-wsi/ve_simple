#!/bin/bash
# Phase 1: everything except FoldX, on GPU, in chunks of proteins.
# Usage:  bash lsf/phase1_gpu.sh all_ids.txt [chunk_size]
set -euo pipefail
IDS=${1:?id list}
CHUNK=${2:-100}
QUEUE=${GPU_QUEUE:-gpu}
mkdir -p lsf/chunks lsf/logs
python -c 'import sys; from ve_simple.utils import read_id_list; print("\n".join(read_id_list(sys.argv[1])))' "$IDS" > lsf/chunks/all_ids.txt
split -l "$CHUNK" -d -a 3 lsf/chunks/all_ids.txt lsf/chunks/ids_
for f in lsf/chunks/ids_*; do
  n=$(basename "$f")
  bsub -q "$QUEUE" -J "ve_${n}" -n 4 -M 32000 -R "select[mem>32000] rusage[mem=32000]" \
       -gpu "num=1:mode=shared" -o "lsf/logs/${n}.out" -e "lsf/logs/${n}.err" \
       "source .venv/bin/activate && ve-simple run --ids $f --out results_phase1/${n} --device cuda --skip foldx --no-thermompnn-download --no-pdf"
done
