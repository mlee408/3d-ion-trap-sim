#!/usr/bin/env bash
# run_top_barriers.sh
#
# Compute transport barriers for a hand-picked list of top-scoring sweep cases.
# Run from inside the sweep output directory (the one containing case_* subdirs).
#
# Usage:
#   bash run_top_barriers.sh 3 7 12 15 21 28 33 37 41 45
#
# Override script location if it is not at ../src/compute_transport_barrier.py:
#   TRANSPORT_SCRIPT=/path/to/compute_transport_barrier.py bash run_top_barriers.sh ...

set -euo pipefail

# ── Locate compute_transport_barrier.py ──────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSPORT_SCRIPT="${TRANSPORT_SCRIPT:-${SCRIPT_DIR}/../src/compute_transport_barrier.py}"

if [[ ! -f "$TRANSPORT_SCRIPT" ]]; then
    echo "ERROR: compute_transport_barrier.py not found at: $TRANSPORT_SCRIPT" >&2
    echo "Set TRANSPORT_SCRIPT=/path/to/compute_transport_barrier.py to override." >&2
    exit 1
fi

# ── Python interpreter (must have dolfinx / mpi4py) ──────────────────────────
# Default: fenicsx conda env. Override with PYTHON=/path/to/python.
PYTHON="${PYTHON:-/Users/michaelee408/Downloads/ENTER/envs/fenicsx/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: python not found at: $PYTHON" >&2
    echo "Set PYTHON=/path/to/python3 to override." >&2
    exit 1
fi

TRANSPORT_SCRIPT="$(realpath "$TRANSPORT_SCRIPT")"
SWEEP_DIR="$(pwd)"
LOG_FILE="${SWEEP_DIR}/transport_barrier_run.log"
SUMMARY_CSV="${SWEEP_DIR}/summary.csv"

if [[ $# -eq 0 ]]; then
    echo "Usage: bash run_top_barriers.sh <case_num> [case_num ...]" >&2
    echo "Example: bash run_top_barriers.sh 3 7 12 15 21 28 33 37 41 45" >&2
    exit 1
fi

echo "=================================================="
echo " Transport barrier computation"
echo " sweep dir : $SWEEP_DIR"
echo " script    : $TRANSPORT_SCRIPT"
echo " log file  : $LOG_FILE"
echo " cases     : $*"
echo "=================================================="
echo ""

# Truncate (not append) log for this run
: > "$LOG_FILE"

t_start=$SECONDS

# ── Per-case tracking arrays ──────────────────────────────────────────────────
declare -a CASES_OK=()
declare -a CASES_SKIPPED=()
declare -a CASES_ERROR=()

# ── Helper: find the JSON report in a case directory ─────────────────────────
find_report() {
    local case_dir="$1"
    # Try common patterns in priority order
    local f
    for f in "$case_dir"/*_sweep.json "$case_dir"/*_report.json "$case_dir"/report.json; do
        # Expand glob manually
        [[ -f "$f" ]] && echo "$f" && return 0
    done
    # Fallback: sole JSON that isn't params.json
    local jsons=()
    while IFS= read -r -d '' j; do
        [[ "$(basename "$j")" != "params.json" ]] && jsons+=("$j")
    done < <(find "$case_dir" -maxdepth 1 -name "*.json" -print0 2>/dev/null)
    if [[ ${#jsons[@]} -eq 1 ]]; then
        echo "${jsons[0]}"
        return 0
    fi
    return 1
}

# ── Process each case ─────────────────────────────────────────────────────────
for num in "$@"; do
    case_id=$(printf "case_%04d" "$num")
    case_dir="${SWEEP_DIR}/${case_id}"

    echo -n "[${case_id}] "

    # Check directory exists
    if [[ ! -d "$case_dir" ]]; then
        echo "SKIP — directory not found: $case_dir"
        CASES_SKIPPED+=("$case_id:no_dir")
        echo "[${case_id}] SKIP — directory not found" >> "$LOG_FILE"
        continue
    fi

    # Check report exists and success==true (or no success field)
    report_path=""
    if report_path=$(find_report "$case_dir"); then
        # Use python to check success field robustly
        success=$("$PYTHON" -c "
import json, sys
try:
    r = json.load(open('$report_path'))
    v = r.get('success', True)
    print('true' if v else 'false')
except Exception as e:
    print('error: ' + str(e), file=sys.stderr)
    print('unknown')
" 2>>"$LOG_FILE")
        if [[ "$success" == "false" ]]; then
            echo "SKIP — report.success == false"
            CASES_SKIPPED+=("$case_id:failed_case")
            echo "[${case_id}] SKIP — report.success == false" >> "$LOG_FILE"
            continue
        fi
    else
        echo "SKIP — no JSON report found in $case_dir"
        CASES_SKIPPED+=("$case_id:no_report")
        echo "[${case_id}] SKIP — no JSON report found" >> "$LOG_FILE"
        continue
    fi

    # Run compute_transport_barrier.py, tee to terminal + log
    echo "running..."
    {
        echo "===== ${case_id} ====="
        echo "  report: $report_path"
        date
    } >> "$LOG_FILE"

    set +e
    "$PYTHON" "$TRANSPORT_SCRIPT" \
        --case-dir "$case_dir" \
        --output-mode patch \
        --overwrite \
        2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ $rc -eq 0 ]]; then
        CASES_OK+=("$case_id")
        echo "[${case_id}] exit 0 (ok)" >> "$LOG_FILE"
    else
        echo "[${case_id}] ERROR — exit code $rc" | tee -a "$LOG_FILE"
        CASES_ERROR+=("$case_id:exit_$rc")
    fi
    echo "" >> "$LOG_FILE"
done

t_elapsed=$(( SECONDS - t_start ))

# ── Per-case status summary ───────────────────────────────────────────────────
echo ""
echo "=================================================="
echo " Run complete  (${t_elapsed}s elapsed)"
echo "  ok      : ${#CASES_OK[@]}"
echo "  skipped : ${#CASES_SKIPPED[@]}"
echo "  error   : ${#CASES_ERROR[@]}"

if [[ ${#CASES_SKIPPED[@]} -gt 0 ]]; then
    echo "  skipped list: ${CASES_SKIPPED[*]}"
fi
if [[ ${#CASES_ERROR[@]} -gt 0 ]]; then
    echo "  error list  : ${CASES_ERROR[*]}"
fi
echo "=================================================="

# ── Barrier summary table ─────────────────────────────────────────────────────
if [[ ${#CASES_OK[@]} -eq 0 ]]; then
    echo ""
    echo "No successful cases — nothing to summarise."
    exit 0
fi

# Build list of case dirs for the ok cases
ok_dirs=()
for cid in "${CASES_OK[@]}"; do
    ok_dirs+=("${SWEEP_DIR}/${cid}")
done

# Python one-liner to extract and print ranked table
"$PYTHON" - "${ok_dirs[@]}" "$SUMMARY_CSV" <<'PYEOF'
import json, csv, sys
from pathlib import Path

case_dirs = []
summary_csv = None
for arg in sys.argv[1:]:
    p = Path(arg)
    if p.is_dir():
        case_dirs.append(p)
    elif p.suffix == ".csv" and p.exists():
        summary_csv = p

# Load scores from summary.csv keyed by case_id
scores = {}
if summary_csv:
    try:
        with summary_csv.open() as f:
            for row in csv.DictReader(f):
                cid = row.get("case_id", "")
                s   = row.get("score", "")
                try:
                    scores[cid] = float(s)
                except (TypeError, ValueError):
                    pass
    except Exception:
        pass

def find_report(d):
    for pat in ["*_sweep.json", "*_report.json", "report.json"]:
        hits = [p for p in d.glob(pat) if p.name != "params.json"]
        if hits:
            return hits[0]
    jsons = [p for p in d.glob("*.json") if p.name != "params.json"]
    return jsons[0] if len(jsons) == 1 else None

rows = []
for cd in case_dirs:
    rp = find_report(cd)
    if rp is None:
        continue
    try:
        r = json.loads(rp.read_text())
    except Exception:
        continue
    barrier = r.get("transport_barrier_xscan_eV")
    if barrier is None:
        continue
    rows.append({
        "case_id":    cd.name,
        "barrier_eV": float(barrier),
        "peak_x_um":  (r.get("transport_xscan_peak_x_m") or 0) * 1e6,
        "junc_um":    (r.get("transport_xscan_nearest_junction_x_m") or 0) * 1e6,
        "reached":    "yes" if r.get("transport_xscan_reached_junction") else "no",
        "score":      scores.get(cd.name),
    })

rows.sort(key=lambda x: x["barrier_eV"])

print()
print(f"{'Rank':<5} {'case_id':<14} {'barrier_meV':>12} {'peak_x_µm':>10} {'junction_µm':>12} {'reached':>8} {'score':>8}")
print("─" * 74)
for i, row in enumerate(rows, 1):
    score_str = f"{row['score']:.4g}" if row["score"] is not None else "—"
    print(
        f"{i:<5} {row['case_id']:<14} "
        f"{row['barrier_eV']*1e3:>12.3f} "
        f"{row['peak_x_um']:>10.1f} "
        f"{row['junc_um']:>12.1f} "
        f"{row['reached']:>8} "
        f"{score_str:>8}"
    )
print()
PYEOF
