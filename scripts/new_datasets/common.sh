# shared helpers for scripts/new_datasets/*.sh (sourced)
set -eu          # no pipefail: `ls | wc -l` on an empty glob is a count of 0, not an error
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
source scripts/new_datasets/config.sh
[ -n "${VIRTUAL_ENV:-}" ] || source .venv/bin/activate
tag_of() { local t; t=$(basename "$1"); echo "${t%.*}"; }
active() { bjobs -noheader -J "$1" 2>/dev/null | grep -qE "RUN|PEND"; }      # job name has queued/running jobs
