#!/usr/bin/env bash
#
# Run the whole Step 2 benchmark grid, once, unattended.
#
# The approved plan forbids a Claude Code session from launching anything over
# ~15 minutes, because a disconnect kills it.  This is the other half of that
# arrangement: one command to start, which then does the right thing in the
# right order and survives logout.
#
#     ./scripts/run_step2_grid.sh --detach     # start it and walk away
#     ./scripts/run_step2_grid.sh --status     # how far has it got?
#
# Safe to re-run.  Every stage underneath skips cells already on disk, so an
# interrupted grid resumes rather than restarting.
#
# Ordering, which is the whole reason this exists:
#
#     validate_against_a157    independent of everything, ~16 s
#     generate_benchmark       BARRIER -- the runners glob for *complete*
#                              worlds, so one started early would quietly
#                              process a subset and look finished
#     run_baselines         ]
#     run_baselines --structural ] in parallel: disjoint output paths
#     sampling_only_experiment   ]
#     score  +  score --structural   in parallel: different CSVs
#     make_step2_figures
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

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

LOGS="$REPO/outputs/step2/logs"
SUMMARY="$LOGS/grid_summary.log"
JOBS="${GN_JOBS:-16}"
BENCH_CONFIG="configs/step2_benchmark.yaml"
BASE_CONFIG="configs/step2_baselines.yaml"
SCENARIOS=()
METHODS=()
REALIZATIONS=""
GEN_ARGS=()
BASE_ARGS=()
DETACH=0
DRYRUN=0
FAILURES=0

# ---------------------------------------------------------------- functions

usage() {
    sed -n '3,27p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'
    cat <<'USAGE'

Options
  --detach            re-exec under setsid and return immediately
  --dry-run           print the stages and their arguments, run nothing
  --status            print every stage's progress and exit
  --jobs N            worker processes per runner (default 16)
  --scenarios A B     restrict generation to these scenarios
  --realizations N    override the per-scenario count (for a smoke run)
  --methods A B       restrict the baselines to these methods
  -h, --help          this
USAGE
}

status() {
    if ! compgen -G "$LOGS/*_progress.json" > /dev/null; then
        echo "no stage has started yet (nothing under $LOGS)"
        return 0
    fi
    "$PYTHON" - "$LOGS" <<'PY'
import json
import pathlib
import sys

logs = pathlib.Path(sys.argv[1])
rows, failures = [], []
for path in sorted(logs.glob("*_progress.json")):
    d = json.loads(path.read_text())
    finished = d["done"] + d["skipped"] + d["failed"]
    eta = d.get("eta_s")
    if finished >= d["total"]:
        state = "done"
    elif eta:
        state = f"~{eta / 60:.0f} min left"
    else:
        state = "running"
    rows.append((
        d["label"],
        f'{finished}/{d["total"]}',
        f'{d["done"]} done, {d["skipped"]} skipped, {d["failed"]} failed',
        f'{d["elapsed_s"] / 60:.1f} min',
        state,
    ))
    failures += [f'{d["label"]}: {f}' for f in d.get("failures", [])]

width = [max(len(r[i]) for r in rows) for i in range(5)]
for row in rows:
    print("  ".join(cell.ljust(w) for cell, w in zip(row, width)))
for failure in failures[:10]:
    print(f"  FAILED {failure}")
if len(failures) > 10:
    print(f"  ... and {len(failures) - 10} more failures")
PY
}

note() {
    if [[ "$DRYRUN" == 1 ]]; then
        printf '%s\n' "$*"
    else
        printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$SUMMARY"
    fi
}

# A stage is a named command whose output goes to its own log.  It never aborts
# the script, so one broken stage cannot cost a night's compute for the stages
# that do not depend on it.  Note it cannot increment FAILURES itself: when
# backgrounded it runs in a subshell and the parent would never see it, so the
# caller counts from the exit code.
stage() {
    local name="$1"; shift
    if [[ "$DRYRUN" == 1 ]]; then
        printf 'would run  %-18s %s\n' "$name" "$*"
        return 0
    fi
    local log="$LOGS/$name.log"
    local start=$SECONDS
    note "START $name  (log: $log)"
    "$@" > "$log" 2>&1
    local code=$?
    local mins=$(( (SECONDS - start) / 60 ))
    if [[ $code -eq 0 ]]; then
        note "  OK   $name  after ${mins} min"
    else
        note "  FAIL $name  exit $code after ${mins} min -- see $log"
    fi
    return $code
}

# ------------------------------------------------------------- arguments

while [[ $# -gt 0 ]]; do
    case "$1" in
        --detach) DETACH=1; shift ;;
        --dry-run) DRYRUN=1; shift ;;
        --status) status; exit 0 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --scenarios)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do SCENARIOS+=("$1"); shift; done ;;
        --realizations) REALIZATIONS="$2"; shift 2 ;;
        --methods)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do METHODS+=("$1"); shift; done ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option $1" >&2; usage; exit 2 ;;
    esac
done

mkdir -p "$LOGS"

# argparse's nargs="*" keeps only the LAST occurrence of a repeated flag, so the
# names must be emitted once as "--scenarios A B C". Collecting them first and
# building the argument list here is the difference between running three
# scenarios and silently running one.
[[ ${#SCENARIOS[@]} -gt 0 ]] && GEN_ARGS+=(--scenarios "${SCENARIOS[@]}")
[[ -n "$REALIZATIONS" ]] && GEN_ARGS+=(--realizations "$REALIZATIONS")
[[ ${#METHODS[@]} -gt 0 ]] && BASE_ARGS+=(--methods "${METHODS[@]}")

if [[ "$DETACH" == 1 ]]; then
    args=(--jobs "$JOBS")
    [[ ${#SCENARIOS[@]} -gt 0 ]] && args+=(--scenarios "${SCENARIOS[@]}")
    [[ -n "$REALIZATIONS" ]] && args+=(--realizations "$REALIZATIONS")
    [[ ${#METHODS[@]} -gt 0 ]] && args+=(--methods "${METHODS[@]}")
    setsid nohup "${BASH_SOURCE[0]}" "${args[@]}" > "$LOGS/grid.log" 2>&1 < /dev/null &
    echo "started, pid $!"
    echo "  log       $LOGS/grid.log"
    echo "  progress  $0 --status"
    exit 0
fi

# ------------------------------------------------------------- preflight

note "=============================================================="
note "Step 2 grid starting"
note "  python  $PYTHON"
note "  jobs    $JOBS"
note "  disk    $(df -h "$REPO" | awk 'NR==2 {print $4" free on "$6}')"

if [[ "$DRYRUN" != 1 ]] && { [[ -z "$PYTHON" ]] || ! "$PYTHON" -c "import gn_next_gen" 2>/dev/null; }; then
    note "FATAL: '$PYTHON' cannot import gn_next_gen. Set GN_PYTHON to the env's python."
    exit 1
fi
avail_gb=$(df -BG --output=avail "$REPO" 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -n "$avail_gb" ]] && (( avail_gb < 20 )); then
    note "FATAL: only ${avail_gb} GB free; the grid writes roughly 3 GB and wants headroom."
    exit 1
fi

# -------------------------------------------- independent of everything

stage a157 "$PYTHON" -u scripts/validate_against_a157.py || FAILURES=$((FAILURES + 1))

# ------------------------------------------------------------- the barrier

if ! stage generate "$PYTHON" -u scripts/generate_benchmark.py \
        --config "$BENCH_CONFIG" --jobs "$JOBS" "${GEN_ARGS[@]+"${GEN_ARGS[@]}"}"; then
    note "ABORTING: every remaining stage reads the worlds this one makes."
    note "Nothing is lost -- re-run this script and it resumes."
    exit 1
fi

# ----------------------------- parallel group: disjoint output paths

note "START parallel group: baselines, structural ensemble, sampling-only"
stage baselines "$PYTHON" -u scripts/run_baselines.py \
    --config "$BASE_CONFIG" --jobs "$JOBS" "${BASE_ARGS[@]+"${BASE_ARGS[@]}"}" &
pid_base=$!
stage structural "$PYTHON" -u scripts/run_baselines.py \
    --config "$BASE_CONFIG" --structural --jobs "$JOBS" "${BASE_ARGS[@]+"${BASE_ARGS[@]}"}" &
pid_struct=$!
stage sampling "$PYTHON" -u scripts/sampling_only_experiment.py &
pid_sampling=$!

wait $pid_base;     base_code=$?
wait $pid_struct;   struct_code=$?
wait $pid_sampling; sampling_code=$?
for code in $base_code $struct_code $sampling_code; do
    (( code == 0 )) || FAILURES=$((FAILURES + 1))
done
note "parallel group finished (baselines=$base_code structural=$struct_code sampling=$sampling_code)"

# ------------------------------ scoring: different CSVs, so both at once

pid_score=""
pid_score_s=""
if (( base_code == 0 )); then
    stage score "$PYTHON" -u scripts/score_benchmark.py --config "$BENCH_CONFIG" &
    pid_score=$!
else
    note "  SKIP score -- the baselines did not finish cleanly"
fi
if (( struct_code == 0 )); then
    stage score_structural "$PYTHON" -u scripts/score_benchmark.py \
        --config "$BENCH_CONFIG" --structural &
    pid_score_s=$!
else
    note "  SKIP score --structural -- the ensemble did not finish cleanly"
fi
if [[ -n "$pid_score" ]]; then wait $pid_score || FAILURES=$((FAILURES + 1)); fi
if [[ -n "$pid_score_s" ]]; then wait $pid_score_s || FAILURES=$((FAILURES + 1)); fi

# ---------------------------------------------------------------- figures

stage figures "$PYTHON" -u scripts/make_step2_figures.py || FAILURES=$((FAILURES + 1))

note "--------------------------------------------------------------"
if (( FAILURES == 0 )); then
    note "Step 2 grid COMPLETE, no stage failed."
else
    note "Step 2 grid finished with $FAILURES failed stage(s) -- see the logs above."
fi
note "Re-running this script resumes; finished cells are skipped."
note ""
[[ "$DRYRUN" == 1 ]] || status | tee -a "$SUMMARY"
exit $(( FAILURES > 0 ))
