#!/bin/bash
# Reviewed (Swiss-Prot) accessions of one or more organisms, one file per taxid, in the same format as
# uniprot_reviewed_9606.txt.  Usage: bash scripts/fetch_proteome_ids.sh OUTDIR TAXID [TAXID ...]
#   e.g. bash scripts/fetch_proteome_ids.sh /lustre/.../retrieve_proteomes 10090 7955 7227 6239 3702 284812 83333
# Common model organisms: 10090 mouse, 10116 rat, 7955 zebrafish, 7227 fly, 6239 worm, 3702 Arabidopsis,
# 284812 fission yeast, 83333 E. coli K-12, 44689 Dictyostelium.
set -euo pipefail
OUT=${1:?output directory}; shift
mkdir -p "$OUT"
for T in "$@"; do
  f="$OUT/uniprot_reviewed_${T}.txt"
  expected=$(curl -sS -D - -o /dev/null "https://rest.uniprot.org/uniprotkb/search?query=organism_id:${T}+AND+reviewed:true&size=1" \
             | tr -d '\r' | awk 'tolower($1)=="x-total-results:"{print $2}')
  curl -sS --retry 5 -o "$f.part" "https://rest.uniprot.org/uniprotkb/stream?query=organism_id:${T}+AND+reviewed:true&format=list"
  got=$(wc -l < "$f.part")
  [ "$got" -ge "$expected" ] || { echo "taxid $T: downloaded $got of $expected accessions; run again"; exit 1; }
  mv "$f.part" "$f"
  echo "taxid $T: $got reviewed accessions -> $f"
done
