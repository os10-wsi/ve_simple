#!/bin/bash
# Stage 3: assemble each dataset whose GPU chunks are all complete (ALLOW_INCOMPLETE=1 to assemble anyway).
source "$(dirname "$0")/common.sh"
for NAME in "${!DATASETS[@]}"; do
  IDS=${DATASETS[$NAME]}; TAG=$(tag_of "$IDS")
  [ -d "lsf/chunks/$TAG" ] || { echo "$NAME: not submitted to the GPU yet (stage 2)"; continue; }
  if [ -f "results_$NAME/summary.json" ] && [ -z "${FORCE:-}" ]; then echo "$NAME: results_$NAME/ done (FORCE=1 to redo)"; continue; fi
  if active "ve_${TAG}_*"; then echo "$NAME: GPU jobs still queued/running"; continue; fi
  n_chunks=$(ls lsf/chunks/$TAG/ids_* | wc -l)
  n_done=$(ls results_phase1/$TAG/*/summary.json 2>/dev/null | wc -l)
  bad=$(ve-simple incomplete-chunks "lsf/chunks/$TAG" 2>/dev/null | wc -l)
  if { [ "$n_done" -lt "$n_chunks" ] || [ "$bad" -gt 0 ]; } && [ -z "${ALLOW_INCOMPLETE:-}" ]; then
    echo "$NAME: $n_done/$n_chunks chunks finished, $bad with missing tables -> run 2_gpu.sh again (or ALLOW_INCOMPLETE=1)"
    continue
  fi
  echo "$NAME: assembling into results_$NAME/"
  bash lsf/phase3_assemble.sh "$IDS" "results_$NAME"
done
