#!/bin/bash
# Resubmit the phase-1 chunks of one ID file that are not complete: chunks without summary.json,
# and chunks containing a protein whose ESM-1v / SaProt / exposure / ThermoMPNN table is missing
# from the cache (e.g. deleted by a clean-up). Proteins already cached are reloaded in seconds.
# Usage:  bash lsf/phase1_resubmit_failed.sh all_ids.txt
set -euo pipefail
IDS=${1:?id list (same file given to phase1_gpu.sh)}
QUEUE=${GPU_QUEUE:-gpu}
MEM=${HOST_MEM:-24000}
TAG=$(basename "$IDS"); TAG=${TAG%.*}
CHUNKS="lsf/chunks/${TAG}"
[ -d "$CHUNKS" ] || { echo "no chunks for $TAG; run phase1_gpu.sh first"; exit 1; }
INCOMPLETE=$(ve-simple incomplete-chunks "$CHUNKS")
n_sub=0
for f in "$CHUNKS"/ids_*; do
  n=$(basename "$f")
  if [ -f "results_phase1/${TAG}/${n}/summary.json" ] && ! grep -qx "$f" <<< "$INCOMPLETE"; then
    continue
  fi
  if bjobs -noheader -J "ve_${TAG}_${n}" 2>/dev/null | grep -qE "RUN|PEND"; then
    echo "${TAG}_${n}: still queued/running, skipping"; continue
  fi
  rm -f "lsf/logs/${TAG}_${n}.log" "lsf/logs/${TAG}_${n}.out" "lsf/logs/${TAG}_${n}.err"
  bsub -q "$QUEUE" -J "ve_${TAG}_${n}" -n 4 -M "$MEM" -R "select[mem>$MEM] rusage[mem=$MEM]" \
       -gpu "${GPU_REQ:-num=1:mode=shared:j_exclusive=yes}" -o "lsf/logs/${TAG}_${n}.log" \
       "source .venv/bin/activate && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && ve-simple run --ids $f --out results_phase1/${TAG}/${n} --device cuda --skip foldx --no-thermompnn-download --cache-only"
  n_sub=$((n_sub + 1))
done
echo "resubmitted ${n_sub} chunk(s) for ${TAG}"
