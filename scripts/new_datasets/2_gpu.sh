#!/bin/bash
# Stage 2: GPU phase 1 for every dataset whose ID file exists. Safe to re-run: datasets already submitted
# get only their missing or incomplete chunks resubmitted.
source "$(dirname "$0")/common.sh"
for NAME in "${!DATASETS[@]}"; do
  IDS=${DATASETS[$NAME]}; TAG=$(tag_of "$IDS")
  if [ ! -f "$IDS" ]; then echo "$NAME: $IDS not there yet (stage 1)"; continue; fi
  if [ -d "lsf/chunks/$TAG" ]; then
    echo "$NAME: already submitted; resubmitting incomplete chunks"
    bash lsf/phase1_resubmit_failed.sh "$IDS"
  else
    echo "$NAME: submitting"
    bash lsf/phase1_gpu.sh "$IDS" 100
  fi
done
