#!/bin/bash
# Stage 5: membrane analysis for every assembled dataset: prepare -> per-protein correlations -> notebooks.
# Jobs wait for each other (and for the shape collect job if it is still queued). Re-run after more datasets finish.
source "$(dirname "$0")/common.sh"
cd membrane_analysis
SHAPE_DEP=""
if active protshape_collect || active "protshape*"; then SHAPE_DEP="protshape_collect"; echo "shape jobs still running: prepare jobs will wait for them"; fi
for NAME in "${!DATASETS[@]}"; do
  [ -f "../results_$NAME/summary.json" ] || { echo "$NAME: results_$NAME/ not assembled yet (stage 3)"; continue; }
  if [ -f "data_$NAME/meta.json" ] && [ -z "${FORCE:-}" ]; then
    echo "$NAME: data_$NAME/ exists (FORCE=1 to rebuild); submitting notebooks only"
    [ -f "data_$NAME/protein_correlations.parquet" ] || bash lsf/protcorr.sh "$NAME"
    AFTER_P=$([ -f "data_$NAME/protein_correlations.parquet" ] && echo "" || echo 1)
  else
    AFTER=$SHAPE_DEP bash lsf/prepare.sh "../results_$NAME"
    AFTER=1 bash lsf/protcorr.sh "$NAME"
    AFTER_P=1
  fi
  if [ "$NAME" = families ]; then
    AFTER=$AFTER_P bash lsf/run_families.sh families
  else
    AFTER=$AFTER_P STRUCT_SETS=$STRUCT_SETS bash lsf/run_talk.sh "$NAME"
    AFTER=$([ -f "data_$NAME/meta.json" ] && echo "" || echo 1) bash lsf/run_notebook.sh "$NAME"
  fi
done
