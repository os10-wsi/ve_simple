#!/bin/bash
# Resubmit only the phase-1 chunks that did not finish (no summary.json in their results dir).
# Usage:  bash lsf/phase1_resubmit_failed.sh        (after the original phase1_gpu.sh submission)
set -euo pipefail
QUEUE=${GPU_QUEUE:-gpu}
n_sub=0
for f in lsf/chunks/ids_*; do
  n=$(basename "$f")
  if [ -f "results_phase1/${n}/summary.json" ]; then
    continue
  fi
  if bjobs -noheader -J "ve_${n}" 2>/dev/null | grep -qE "RUN|PEND"; then
    echo "${n}: still queued/running, skipping"
    continue
  fi
  mv -f "lsf/logs/${n}.err" "lsf/logs/${n}.err.prev" 2>/dev/null || true
  mv -f "lsf/logs/${n}.out" "lsf/logs/${n}.out.prev" 2>/dev/null || true
  bsub -q "$QUEUE" -J "ve_${n}" -n 4 -M 12000 -R "select[mem>12000] rusage[mem=12000]" \
       -gpu "${GPU_REQ:-num=1:mode=shared:j_exclusive=yes}" -o "lsf/logs/${n}.out" -e "lsf/logs/${n}.err" \
       "source .venv/bin/activate && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && ve-simple run --ids $f --out results_phase1/${n} --device cuda --skip foldx --no-thermompnn-download --no-pdf"
  n_sub=$((n_sub + 1))
done
echo "resubmitted ${n_sub} chunk(s)"
