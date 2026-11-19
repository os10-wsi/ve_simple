#!/bin/bash
# Stage 1: download proteome ID lists and submit the membrane-family selection (MMseqs2).
source "$(dirname "$0")/common.sh"
bash scripts/fetch_proteome_ids.sh "$P" $TAXIDS
if [ -f sets/families/all_family_ids.txt ] && [ -z "${FORCE:-}" ]; then
  echo "families: sets/families/all_family_ids.txt exists ($(wc -l < sets/families/all_family_ids.txt) accessions); FORCE=1 to rerun"
elif command -v "$MMSEQS" >/dev/null 2>&1 || [ -x "$MMSEQS" ]; then
  MMSEQS=$MMSEQS bash lsf/family_sets.sh "$FAMILY_N" "$FAMILY_THREADS"
else
  echo "families: MMseqs2 not found ($MMSEQS). Install it (see docs/new_datasets.md) and set MMSEQS=/path/to/mmseqs"
fi
