#!/usr/bin/env bash
# Step 3, unattended, in the only order that is correct.
#
# The PI launches this, not a Claude Code session (Step 2 decision 10, carried
# forward): the benchmark cells and the real-data runs are hours, and a session
# dies on a disconnect.  Every stage is resumable, so an interrupted grid
# continues rather than restarting.
#
#   ./scripts/run_step3_grid.sh --detach     # returns at once, survives logout
#   ./scripts/run_step3_grid.sh --status     # what each stage has done so far
#
# One dependency is hard and the script enforces it: nothing that *reads* a fit
# may start before the fit exists.  Everything else is independent.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CONFIG="configs/step3_network.yaml"
# Memory, not cores, caps these.  A worker peaks near 10 GB during the Hessian
# (see inference/composite.HESSIAN_CHUNK), so 6 workers is about 60 GB and 16
# would be 160 GB.  Raise it only after checking free memory.
JOBS=4
BENCH_JOBS=6
LOGS="outputs/step3/logs"
RUN_ID="${GN_RUN_ID:-$$}"
export GN_RUN_ID="$RUN_ID"
DETACH=0
STATUS=0
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach) DETACH=1; shift ;;
    --status) STATUS=1; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --bench-jobs) BENCH_JOBS="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      echo
      echo "Stages: graph, benchmark, runs, figures.  --only NAME runs one."
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOGS"

if [[ "$STATUS" == "1" ]]; then
  echo "=== Step 3 status (run id $RUN_ID) ==="
  for progress in outputs/step3/runs/progress.json outputs/step2/baselines/progress.json; do
    [[ -f "$progress" ]] && { echo "--- $progress"; python - "$progress" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1]).read())
mine = str(data.get("run_id", "")) == str(__import__("os").environ.get("GN_RUN_ID", ""))
tag = "" if mine else "   (other run)"
print(f"  {data.get('label','')}: {data.get('done',0)} done, "
      f"{data.get('skipped',0)} skipped, {data.get('failed',0)} failed "
      f"of {data.get('total','?')}{tag}")
for line in data.get("failures", [])[-5:]:
    print(f"    ! {line}")
PY
    }
  done
  echo "--- graph"
  [[ -f outputs/step3/graph/summary.json ]] && python -c "
import json; d=json.load(open('outputs/step3/graph/summary.json'))
print('  ', {k: d[k] for k in ('n_stations','n_edges','n_components','n_fingerprint_only')})" \
    || echo "   not run"
  echo "--- figures"
  ls docs/results/figures/step-3/*_light.png 2>/dev/null | wc -l | xargs -I{} echo "   {} light figures"
  exit 0
fi

if [[ "$DETACH" == "1" ]]; then
  DETACH=0
  nohup "$0" --config "$CONFIG" --jobs "$JOBS" --bench-jobs "$BENCH_JOBS" \
        ${ONLY:+--only "$ONLY"} > "$LOGS/grid.log" 2>&1 &
  echo "launched: run id $RUN_ID, pid $!, log $LOGS/grid.log"
  echo "watch with: $0 --status"
  exit 0
fi

FAILED=()
stage() {
  local name="$1"; shift
  if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then
    echo "[$(date +%H:%M:%S)] skipping $name (--only $ONLY)"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] === $name ==="
  if "$@"; then
    echo "[$(date +%H:%M:%S)] $name ok"
  else
    echo "[$(date +%H:%M:%S)] $name FAILED (exit $?)"
    FAILED+=("$name")
  fi
}

# 1. Graph diagnostics.  Minutes, needs no fit, and f29/f39 come from it.
stage graph python -u scripts/graph_diagnostics.py --config "$CONFIG"

# 2. The benchmark cells, through Step 2's own runner -- the integration is one
#    flag, because the scorer discovers cells by glob and needs no change.
stage benchmark python -u scripts/run_baselines.py \
      --config configs/step2_baselines.yaml \
      --methods NetCal NetCal-M --jobs "$BENCH_JOBS"

# 3. The real-data runs: both released anchorings plus the sensitivities.
stage runs python -u scripts/run_network_calibration.py \
      --config "$CONFIG" --jobs "$JOBS"

# 4. Re-score, so Step 3's cells land in the same tables as Step 2's eleven.
stage score python -u scripts/score_benchmark.py

# 5. Figures last; it skips with a reason rather than failing.
stage figures python -u scripts/make_step3_figures.py

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "=== ${#FAILED[@]} stage(s) failed: ${FAILED[*]} ==="
  exit 1
fi
echo "=== all stages ok ==="
