#!/bin/bash
# Phase 3: assemble cached scores into result tables, AUC statistics and the ROC PDF.
# Runs as a job ARRAY: each element assembles one chunk of proteins (default 100) into
# $OUT/parts/ids_NNN, and a final merge job (started when the array has ended) concatenates
# the parts into $OUT.  Re-running the same command resubmits only the unfinished chunks.
#
# Usage:  bash lsf/phase3_assemble.sh all_ids.txt results_final [chunk_size]
# Env:    CPU_QUEUE (normal), MAXPAR (max simultaneous elements, 200), MEM (MB per element, 16000),
#         SQLITE=1 to also write a per-part SQLite file (off: it is ~3x the TSVs and not used downstream).
set -euo pipefail
IDS=${1:?id list}
OUT=${2:-results_final}
CHUNK=${3:-100}
QUEUE=${CPU_QUEUE:-normal}
MAXPAR=${MAXPAR:-200}
MEM=${MEM:-16000}
TAG=$(basename "$IDS"); TAG=${TAG%.*}
CHUNKS="lsf/chunks_assemble/${TAG}"
PARTS="$OUT/parts"
mkdir -p lsf/logs "$PARTS"
SQLITE_FLAG=$([ "${SQLITE:-0}" = "1" ] && echo "" || echo "--no-sqlite")

if [ ! -f "$CHUNKS/.chunk_size_$CHUNK" ]; then
  rm -rf "$CHUNKS"; mkdir -p "$CHUNKS"
  python -c 'import sys; from ve_simple.utils import read_id_list; print("\n".join(read_id_list(sys.argv[1])))' "$IDS" > "$CHUNKS/all_ids.txt"
  split -l "$CHUNK" -d -a 3 "$CHUNKS/all_ids.txt" "$CHUNKS/ids_"
  touch "$CHUNKS/.chunk_size_$CHUNK"
fi
N=$(ls "$CHUNKS"/ids_* | wc -l)

# array indices (1-based) of chunks without a finished part
TODO=()
for i in $(seq 1 "$N"); do
  n=$(printf "ids_%03d" $((i - 1)))
  [ -f "$PARTS/$n/summary.json" ] || TODO+=("$i")
done
echo "$(wc -l < "$CHUNKS/all_ids.txt") accessions in $N chunks; ${#TODO[@]} to run (tag: $TAG)"

DEP=""
if [ ${#TODO[@]} -gt 0 ]; then
  IDX=$(IFS=,; echo "${TODO[*]}")
  bsub -q "$QUEUE" -J "ve_asm_${TAG}[${IDX}]%${MAXPAR}" -n 1 -M "$MEM" -R "select[mem>$MEM] rusage[mem=$MEM]" \
       -o "lsf/logs/assemble_${TAG}_%I.log" \
       "source .venv/bin/activate && n=\$(printf 'ids_%03d' \$((LSB_JOBINDEX - 1))) && ve-simple run --ids $CHUNKS/\$n --out $PARTS/\$n --skip esm1v,saprot,foldx,thermompnn --no-thermompnn-download --bootstrap 2000 --no-pdf $SQLITE_FLAG"
  DEP="-w ended(ve_asm_${TAG})"
fi
bsub -q "$QUEUE" -J "ve_merge_${TAG}" $DEP -n 1 -M 16000 -R "select[mem>16000] rusage[mem=16000]" \
     -o "lsf/logs/merge_${TAG}.log" \
     "source .venv/bin/activate && ve-simple merge-parts --parts $PARTS --out $OUT"
