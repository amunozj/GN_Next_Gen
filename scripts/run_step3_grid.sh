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

# The environment's interpreter, resolved explicitly.  A bare `python` is whatever
# is on PATH, and a detached launch from a shell that never ran
# `conda activate gn-next-gen` gets a system python with no scipy -- which is
# exactly how the first launch of this script died, five stages in five seconds.
# run_step2_grid.sh already did it this way; this one did not, and should have.
PYTHON="${GN_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    for candidate in \
        "$HOME/.conda/envs/gn-next-gen/bin/python" \
        "/d0/amunozj/.conda/envs/gn-next-gen/bin/python" \
        "$(command -v python3 || true)"
    do
        [[ -x "$candidate" ]] && PYTHON="$candidate" && break
    done
fi

CONFIG="configs/step3_network.yaml"
# Cores cap these, not memory -- a correction to this file's first version, which
# blamed memory. A worker peaks near 10 GB during the Hessian, which on this
# machine's 1 TB is nothing; what it does use is about 8.5 cores (OMP_NUM_THREADS
# below), so six workers already occupy ~51 of 64. Measured: six workers, ~13 min
# per benchmark cell. Raising this past 8 oversubscribes and buys nothing.
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

# Every stage runs JAX. JAX takes the GPU by default and preallocates most of its
# memory, so parallel workers cannot share one: the first real grid lost 101 of
# 102 benchmark cells to CUDA_ERROR_OUT_OF_MEMORY while reporting success. CPU is
# also faster here (Step 1 measured the GPU 4.8x slower for a sequential scan).
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export GN_DEVICE="${GN_DEVICE:-cpu}"
# Six JAX workers each spawning a thread per core is how a 64-core box reaches a
# load average of 147. Give each worker a slice.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ "$STATUS" == "1" ]]; then
  echo "=== Step 3 status (run id $RUN_ID) ==="
  for progress in outputs/step3/runs/progress.json outputs/step2/logs/baselines_progress.json; do
    [[ -f "$progress" ]] && { echo "--- $progress"; "$PYTHON" - "$progress" <<'PY'
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
  [[ -f outputs/step3/graph/summary.json ]] && "$PYTHON" -c "
import json; d=json.load(open('outputs/step3/graph/summary.json'))
print('  ', {k: d[k] for k in ('n_stations','n_edges','n_components','n_fingerprint_only')})" \
    || echo "   not run"
  echo "--- figures"
  ls docs/results/figures/step-3/*_light.png 2>/dev/null | wc -l | xargs -I{} echo "   {} light figures"
  exit 0
fi

if [[ "$DETACH" == "1" ]]; then
  DETACH=0
  # `setsid` as well as `nohup`, and it is not belt-and-braces. nohup only blocks
  # SIGHUP; it does not move the job out of the launching shell's session or
  # process group, so when that session is torn down -- a terminal closing, a
  # container stopping, an agent process exiting -- the whole group goes with it.
  # That is how the second launch of this grid died eight minutes in, after the
  # graph stage, with zero cells written and nothing in the log to say why.
  # setsid gives the child its own session, so it has no controlling terminal to
  # lose.
  launcher=()
  if command -v setsid >/dev/null 2>&1; then
    launcher=(setsid)
  else
    echo "note: setsid not found; the run may not survive this shell closing"
  fi
  # Keep the previous log. `> grid.log` truncated it, and the post-mortem of a run
  # that dies without a message is the one time the old log is the only evidence
  # there is -- which is exactly the situation it was destroyed in.
  if [[ -s "$LOGS/grid.log" ]]; then
    mv "$LOGS/grid.log" "$LOGS/grid-$(date +%Y%m%d-%H%M%S).log"
  fi
  GN_PYTHON="$PYTHON" "${launcher[@]}" nohup "$0" --config "$CONFIG" \
        --jobs "$JOBS" --bench-jobs "$BENCH_JOBS" ${ONLY:+--only "$ONLY"} \
        > "$LOGS/grid.log" 2>&1 < /dev/null &
  disown 2>/dev/null || true
  echo "launched: run id $RUN_ID, pid $!, log $LOGS/grid.log"
  echo "watch with: $0 --status"
  echo "it now has its own session, so closing this shell will not stop it"
  exit 0
fi

echo "=============================================================="
echo "Step 3 grid starting"
echo "  run id  $RUN_ID"
echo "  python  ${PYTHON:-<none found>}"
echo "  jobs    runs=$JOBS benchmark=$BENCH_JOBS"
echo "  disk    $(df -h "$REPO" | awk 'NR==2 {print $4" free on "$6}')"

if [[ -z "$PYTHON" ]] || ! "$PYTHON" -c "import gn_next_gen" 2>/dev/null; then
  echo "FATAL: '${PYTHON:-<none>}' cannot import gn_next_gen."
  echo "       Set GN_PYTHON to the environment's python, or activate it first:"
  echo "         conda activate gn-next-gen"
  exit 1
fi
if ! "$PYTHON" -c "import scipy, jax, pandas" 2>/dev/null; then
  echo "FATAL: '$PYTHON' is missing scipy/jax/pandas; it is not the gn-next-gen env."
  exit 1
fi

FAILED=()
stage() {
  local name="$1"; shift
  if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then
    echo "[$(date +%H:%M:%S)] skipping $name (--only $ONLY)"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] === $name ==="
  # Capture the status before `if` consumes it: the first version read `$?`
  # inside the else branch and faithfully reported every failure as "exit 0".
  local status=0
  "$@" || status=$?
  local recorded
  recorded="$(stage_failures "$name")"
  if [[ "$status" -eq 0 && "${recorded:-0}" -eq 0 ]]; then
    echo "[$(date +%H:%M:%S)] $name ok"
  elif [[ "$status" -eq 0 ]]; then
    echo "[$(date +%H:%M:%S)] $name FAILED: exit 0 but $recorded unit(s) failed"
    FAILED+=("$name")
  else
    echo "[$(date +%H:%M:%S)] $name FAILED (exit $status)"
    FAILED+=("$name")
  fi
}

# How many units a stage recorded as failed, from whichever progress file it owns.
# An exit code is not enough: run_baselines.py collects per-cell failures and
# still exits 0, so the first real grid printed "benchmark ok" after losing 101
# of 102 cells.
stage_failures() {
  local name="$1" progress=""
  case "$name" in
    benchmark) progress="outputs/step2/logs/baselines_progress.json" ;;
    runs)      progress="outputs/step3/runs/progress.json" ;;
    *)         echo 0; return 0 ;;
  esac
  if [[ ! -f "$progress" ]]; then echo 0; return 0; fi
  "$PYTHON" -c "import json,sys
try:
    print(int(json.load(open(sys.argv[1])).get('failed', 0)))
except Exception:
    print(0)" "$progress" 2>/dev/null || echo 0
}


# 1. Graph diagnostics.  Minutes, needs no fit, and f29/f39 come from it.
stage graph "$PYTHON" -u scripts/graph_diagnostics.py --config "$CONFIG"

# 2. The benchmark cells, through Step 2's own runner -- the integration is one
#    flag, because the scorer discovers cells by glob and needs no change.
stage benchmark "$PYTHON" -u scripts/run_baselines.py \
      --config configs/step2_baselines.yaml \
      --methods NetCal NetCal-M --jobs "$BENCH_JOBS"

# 3. The real-data runs: both released anchorings plus the sensitivities.
stage runs "$PYTHON" -u scripts/run_network_calibration.py \
      --config "$CONFIG" --jobs "$JOBS"

# 4. Re-score, so Step 3's cells land in the same tables as Step 2's eleven.
stage score "$PYTHON" -u scripts/score_benchmark.py

# 5. Figures last; it skips with a reason rather than failing.
stage figures "$PYTHON" -u scripts/make_step3_figures.py

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "=== ${#FAILED[@]} stage(s) failed: ${FAILED[*]} ==="
  exit 1
fi
echo "=== all stages ok ==="
