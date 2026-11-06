#!/bin/bash
# Phase 1: everything except FoldX, on GPU, in chunks of proteins.
# Usage:  bash lsf/phase1_gpu.sh all_ids.txt [chunk_size]
# Chunks, logs, job names and results are namespaced by the ID file's basename, so several
# proteomes can be submitted independently. The GPU is requested for this job alone
# (j_exclusive=yes): the six models need ~8 GB and a shared card can run out of memory.
# Override with GPU_REQ=..., HOST_MEM=<MB>, GPU_QUEUE=... if your cluster differs.
set -euo pipefail
IDS=${1:?id list}
CHUNK=${2:-100}
QUEUE=${GPU_QUEUE:-gpu}
MEM=${HOST_MEM:-24000}
TAG=$(basename "$IDS"); TAG=${TAG%.*}
CHUNKS="lsf/chunks/${TAG}"
rm -rf "$CHUNKS"; mkdir -p "$CHUNKS" lsf/logs
python -c 'import sys; from ve_simple.utils import read_id_list; print("\n".join(read_id_list(sys.argv[1])))' "$IDS" > "$CHUNKS/all_ids.txt"
split -l "$CHUNK" -d -a 3 "$CHUNKS/all_ids.txt" "$CHUNKS/ids_"
echo "$(wc -l < "$CHUNKS/all_ids.txt") accessions in $(ls "$CHUNKS"/ids_* | wc -l) chunks (tag: $TAG)"
for f in "$CHUNKS"/ids_*; do
  n=$(basename "$f")
  bsub -q "$QUEUE" -J "ve_${TAG}_${n}" -n 4 -M "$MEM" -R "select[mem>$MEM] rusage[mem=$MEM]" \
       -gpu "${GPU_REQ:-num=1:mode=shared:j_exclusive=yes}" -o "lsf/logs/${TAG}_${n}.out" -e "lsf/logs/${TAG}_${n}.err" \
       "source .venv/bin/activate && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && ve-simple run --ids $f --out results_phase1/${TAG}/${n} --device cuda --skip foldx --no-thermompnn-download --no-pdf"
done
