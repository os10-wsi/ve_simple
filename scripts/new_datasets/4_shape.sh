#!/bin/bash
# Stage 4: shape metrics for every cached protein (run once all stage-2 GPU jobs have finished).
# Submits: annotation job -> compute array (only new proteins are computed) -> collect + classify.
# The human/yeast findings.md is left as it is (no `protshape analyze`).
# LOCAL_PREP=1 runs the annotation step here instead (if compute nodes cannot reach ted.cathdb.info).
source "$(dirname "$0")/common.sh"
NSHARDS=${NSHARDS:-50}; WORKERS=${WORKERS:-8}; Q=${CPU_QUEUE:-normal}
for NAME in "${!DATASETS[@]}"; do
  TAG=$(tag_of "${DATASETS[$NAME]}")
  if active "ve_${TAG}_*"; then echo "$NAME: GPU jobs still running; run this stage when they have finished"; exit 1; fi
done
cd shape_analysis; mkdir -p lsf/logs
PREP="protshape index-cache --cache ../cache && protshape annotate --threads 4 && protshape annotate --threads 2 && protshape classify"
if [ -n "${LOCAL_PREP:-}" ]; then
  eval "$PREP"; DEP=""
else
  bsub -q "$Q" -J protshape_prep -n 4 -M 16000 -R "select[mem>16000] rusage[mem=16000]" -o lsf/logs/prep.log \
       "source ../.venv/bin/activate && $PREP"
  DEP="-w done(protshape_prep)"
fi
bsub -q "$Q" -J "protshape[1-${NSHARDS}]" $DEP -n "$WORKERS" -M 16000 -R "select[mem>16000] rusage[mem=16000] span[hosts=1]" \
     -o "lsf/logs/compute_%I.log" \
     "source ../.venv/bin/activate && protshape compute --shard \$((LSB_JOBINDEX - 1)) --nshards $NSHARDS --workers $WORKERS"
bsub -q "$Q" -J protshape_collect -w "ended(protshape)" -n 4 -M 32000 -R "select[mem>32000] rusage[mem=32000]" \
     -o lsf/logs/collect.log "source ../.venv/bin/activate && protshape collect && protshape classify"
echo "submitted shape jobs; logs in shape_analysis/lsf/logs/{prep,compute_*,collect}.log"
