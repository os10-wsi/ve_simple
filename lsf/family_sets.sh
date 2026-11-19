#!/bin/bash
# Pick representative members of the membrane Pfam families with MMseqs2 (one CPU job; needs network).
# Usage: bash lsf/family_sets.sh [N_PER_FAMILY] [THREADS]      (run from the repository root)
# Env:   MMSEQS=/path/to/mmseqs (default: mmseqs on PATH), FAMILIES=sets/families/families.tsv
set -euo pipefail
N=${1:-500}
THREADS=${2:-16}
mkdir -p lsf/logs
bsub -q "${CPU_QUEUE:-normal}" -J family_sets -n "$THREADS" -M 64000 -R "select[mem>64000] rusage[mem=64000] span[hosts=1]" \
     -o lsf/logs/family_sets.log \
     "source .venv/bin/activate && python scripts/family_sets.py --families ${FAMILIES:-sets/families/families.tsv} --n $N --threads $THREADS --mmseqs ${MMSEQS:-mmseqs}"
echo "submitted; log lsf/logs/family_sets.log; outputs in sets/families/"
