#!/bin/bash
# Stage 0 on the cluster: index the ve_simple cache, annotate, classify, pick 200 proteins
# (50 MULTIPASS + 50 SOLUBLE per organism), compute them plus the sanity set, then STOP for review.
# Run from shape_analysis/ with the venv active. Needs network for TED (ted.cathdb.info).
set -euo pipefail
P=/lustre/scratch125/gengen/projects_v2/mp_stability/retrieve_proteomes
protshape index-cache --cache ../cache --ids "$P/uniprot_reviewed_9606.txt" "$P/uniprot_reviewed_559292.txt"
protshape annotate --threads 4
protshape annotate --threads 2      # second pass retries only TED queries that failed
protshape classify
protshape select-stage0 --n 200
python -c "from protshape.sanity import SANITY_IDS; open('results/sanity_ids.txt','w').write('\n'.join(SANITY_IDS)+'\n')"
protshape compute --ids results/stage0_ids.txt results/sanity_ids.txt --workers "${WORKERS:-8}"
protshape collect
protshape classify
protshape sanity || true
protshape analyze
echo "Stage 0 done: review results/sanity.tsv and results/findings.md before the full run."
