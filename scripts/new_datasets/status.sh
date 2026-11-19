#!/bin/bash
# Where every dataset stands.
source "$(dirname "$0")/common.sh"
printf "%-10s %-9s %-12s %-10s %-10s %-9s %s\n" dataset ids gpu_chunks assembled data protcorr figures
for NAME in "${!DATASETS[@]}"; do
  IDS=${DATASETS[$NAME]}; TAG=$(tag_of "$IDS")
  ids=$([ -f "$IDS" ] && wc -l < "$IDS" || echo -)
  if [ -d "lsf/chunks/$TAG" ]; then
    chunks="$(ls results_phase1/$TAG/*/summary.json 2>/dev/null | wc -l)/$(ls lsf/chunks/$TAG/ids_* | wc -l)"
    active "ve_${TAG}_*" && chunks="$chunks*"
  else chunks=-; fi
  asm=$([ -f "results_$NAME/summary.json" ] && echo yes || { { active "ve_asm_$TAG" || active "ve_merge_$TAG"; } && echo running || echo -; })
  data=$([ -f "membrane_analysis/data_$NAME/meta.json" ] && echo yes || echo -)
  pc=$([ -f "membrane_analysis/data_$NAME/protein_correlations.parquet" ] && echo yes || echo -)
  fig=$(ls -d membrane_analysis/figures_talk_$NAME membrane_analysis/figures_$NAME 2>/dev/null | head -1)
  printf "%-10s %-9s %-12s %-10s %-10s %-9s %s\n" "$NAME" "$ids" "$chunks" "$asm" "$data" "$pc" "${fig:--}"
done
echo "(* = GPU jobs still queued or running)"
echo; echo "file quota:"; lfs quota -h -u "${USER:-$(whoami)}" /lustre/scratch125 2>/dev/null | tail -n +2 || true
