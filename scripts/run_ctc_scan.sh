#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_ctc_scan.sh
#
# Run the string-method CTC (Critical Transport Coordinate) path scan to
# compute the transport barrier for one or more sweep cases.  Uses
# --transport-mode string in compute_transport_barrier.py, which traces the
# true lowest-energy path through the junction rather than scanning along a
# fixed axis.
#
# Usage
# -----
#   # All cases in a sweep directory (run from inside it):
#   cd runs/sweeps/n4_thickness_height_sweep_yb2025
#   bash /path/to/run_ctc_scan.sh
#
#   # Specific case numbers from inside a sweep directory:
#   bash run_ctc_scan.sh 3 7 12 35
#
#   # Explicit sweep directory + case numbers:
#   bash run_ctc_scan.sh --sweep-dir runs/sweeps/n4_... 3 7 12
#
#   # Single case directory:
#   bash run_ctc_scan.sh --case-dir runs/sweeps/.../case_0035
#
# Environment overrides
# ---------------------
#   PYTHON=/path/to/python3        (default: fenicsx conda env)
#   TRANSPORT_SCRIPT=/path/to/...  (default: ../src/compute_transport_barrier.py)
#   N_NODES=40                     (CTC string nodes, default: 40)
#   JUNCTION_PITCH=600e-6          (metres, default: 600 µm)
#   OVERWRITE=1                    (re-run even if CTC barrier already present)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Locate compute_transport_barrier.py ──────────────────────────────────────
TRANSPORT_SCRIPT="${TRANSPORT_SCRIPT:-${SCRIPT_DIR}/../src/compute_transport_barrier.py}"
if [[ ! -f "$TRANSPORT_SCRIPT" ]]; then
    echo "ERROR: compute_transport_barrier.py not found at: $TRANSPORT_SCRIPT" >&2
    echo "Set TRANSPORT_SCRIPT=/path/to/compute_transport_barrier.py to override." >&2
    exit 1
fi
TRANSPORT_SCRIPT="$(realpath "$TRANSPORT_SCRIPT")"

# ── Python interpreter ────────────────────────────────────────────────────────
PYTHON="${PYTHON:-/Users/michaelee408/Downloads/ENTER/envs/fenicsx/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: python not found at: $PYTHON" >&2
    echo "Set PYTHON=/path/to/python3 to override." >&2
    exit 1
fi

# ── CTC parameters ────────────────────────────────────────────────────────────
N_NODES="${N_NODES:-40}"
JUNCTION_PITCH="${JUNCTION_PITCH:-600e-6}"
OVERWRITE_FLAG=""
[[ "${OVERWRITE:-0}" == "1" ]] && OVERWRITE_FLAG="--overwrite"

# ── Parse arguments ───────────────────────────────────────────────────────────
SWEEP_DIR=""
CASE_DIR_SINGLE=""
CASE_NUMS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sweep-dir)
            SWEEP_DIR="$(realpath "$2")"
            shift 2
            ;;
        --case-dir)
            CASE_DIR_SINGLE="$(realpath "$2")"
            shift 2
            ;;
        --overwrite)
            OVERWRITE_FLAG="--overwrite"
            shift
            ;;
        --n-nodes)
            N_NODES="$2"
            shift 2
            ;;
        --junction-pitch)
            JUNCTION_PITCH="$2"
            shift 2
            ;;
        --help|-h)
            sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        [0-9]*)
            CASE_NUMS+=("$1")
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ── Determine mode ────────────────────────────────────────────────────────────
if [[ -n "$CASE_DIR_SINGLE" ]]; then
    MODE="single"
elif [[ -n "$SWEEP_DIR" ]]; then
    MODE="sweep"
elif [[ -f "automation_config.json" ]] || ls case_* &>/dev/null 2>&1; then
    # Running from inside a sweep directory
    SWEEP_DIR="$(pwd)"
    MODE="sweep"
else
    echo "ERROR: Could not determine target. Run from a sweep directory, or pass" >&2
    echo "  --sweep-dir DIR, --case-dir DIR, or case numbers as arguments." >&2
    exit 1
fi

# ── Setup log ────────────────────────────────────────────────────────────────
if [[ "$MODE" == "single" ]]; then
    LOG_FILE="${CASE_DIR_SINGLE}/ctc_scan.log"
    SUMMARY_CSV=""
else
    LOG_FILE="${SWEEP_DIR}/ctc_scan.log"
    SUMMARY_CSV="${SWEEP_DIR}/ctc_barriers.csv"
fi

: > "$LOG_FILE"

echo "=================================================="
echo " CTC Transport Barrier Scan (string method)"
echo " mode          : $MODE"
[[ -n "$SWEEP_DIR"        ]] && echo " sweep dir     : $SWEEP_DIR"
[[ -n "$CASE_DIR_SINGLE"  ]] && echo " case dir      : $CASE_DIR_SINGLE"
[[ ${#CASE_NUMS[@]} -gt 0 ]] && echo " case numbers  : ${CASE_NUMS[*]}"
echo " n_nodes       : $N_NODES"
echo " junction pitch: $JUNCTION_PITCH m"
echo " overwrite     : ${OVERWRITE_FLAG:-(no)}"
echo " script        : $TRANSPORT_SCRIPT"
echo " log file      : $LOG_FILE"
echo "=================================================="
echo ""

t_start=$SECONDS

# ── Helper: find JSON report in a case directory ──────────────────────────────
find_report() {
    local case_dir="$1"
    local f
    for f in "$case_dir"/*_sweep.json "$case_dir"/*_report.json "$case_dir"/report.json; do
        [[ -f "$f" ]] && echo "$f" && return 0
    done
    local jsons=()
    while IFS= read -r -d '' j; do
        [[ "$(basename "$j")" != "params.json" ]] && jsons+=("$j")
    done < <(find "$case_dir" -maxdepth 1 -name "*.json" -print0 2>/dev/null)
    [[ ${#jsons[@]} -eq 1 ]] && echo "${jsons[0]}" && return 0
    return 1
}

# ── Helper: run CTC scan on one case directory ────────────────────────────────
declare -a CASES_OK=()
declare -a CASES_SKIPPED=()
declare -a CASES_ERROR=()

run_ctc_case() {
    local case_dir="$1"
    local case_id
    case_id="$(basename "$case_dir")"

    echo -n "[${case_id}] "

    if [[ ! -d "$case_dir" ]]; then
        echo "SKIP — directory not found: $case_dir"
        CASES_SKIPPED+=("$case_id:no_dir")
        echo "[${case_id}] SKIP — directory not found" >> "$LOG_FILE"
        return
    fi

    local report_path=""
    if report_path=$(find_report "$case_dir"); then
        local success
        success=$("$PYTHON" -c "
import json, sys
try:
    r = json.load(open('$report_path'))
    print('true' if r.get('success', True) else 'false')
except Exception as e:
    print('error: ' + str(e), file=sys.stderr)
    print('unknown')
" 2>>"$LOG_FILE")
        if [[ "$success" == "false" ]]; then
            echo "SKIP — report.success == false"
            CASES_SKIPPED+=("$case_id:failed_case")
            echo "[${case_id}] SKIP — report.success == false" >> "$LOG_FILE"
            return
        fi
    else
        echo "SKIP — no JSON report found"
        CASES_SKIPPED+=("$case_id:no_report")
        echo "[${case_id}] SKIP — no JSON report found" >> "$LOG_FILE"
        return
    fi

    echo "running CTC scan..."
    {
        echo "===== ${case_id} ====="
        echo "  case dir: $case_dir"
        echo "  report: $report_path"
        date
    } >> "$LOG_FILE"

    set +e
    "$PYTHON" "$TRANSPORT_SCRIPT" \
        --case-dir "$case_dir" \
        --transport-mode string \
        --junction-pitch "$JUNCTION_PITCH" \
        --n-nodes "$N_NODES" \
        --output-mode patch \
        $OVERWRITE_FLAG \
        2>&1 | tee -a "$LOG_FILE"
    local rc=${PIPESTATUS[0]}
    set -e

    if [[ $rc -eq 0 ]]; then
        CASES_OK+=("$case_id")
        echo "[${case_id}] exit 0 (ok)" >> "$LOG_FILE"
    else
        echo "[${case_id}] ERROR — exit code $rc" | tee -a "$LOG_FILE"
        CASES_ERROR+=("$case_id:exit_$rc")
    fi
    echo "" >> "$LOG_FILE"
}

# ── Execute ───────────────────────────────────────────────────────────────────
if [[ "$MODE" == "single" ]]; then
    run_ctc_case "$CASE_DIR_SINGLE"

elif [[ "$MODE" == "sweep" ]]; then
    if [[ ${#CASE_NUMS[@]} -gt 0 ]]; then
        # Specific case numbers
        for num in "${CASE_NUMS[@]}"; do
            case_id=$(printf "case_%04d" "$num")
            run_ctc_case "${SWEEP_DIR}/${case_id}"
        done
    else
        # All case_* directories
        shopt -s nullglob
        case_dirs=("${SWEEP_DIR}"/case_*)
        shopt -u nullglob
        if [[ ${#case_dirs[@]} -eq 0 ]]; then
            echo "ERROR: no case_* directories found in $SWEEP_DIR" >&2
            exit 1
        fi
        echo "Found ${#case_dirs[@]} cases in $SWEEP_DIR"
        echo ""
        for cd in "${case_dirs[@]}"; do
            [[ -d "$cd" ]] && run_ctc_case "$cd"
        done
    fi
fi

t_elapsed=$(( SECONDS - t_start ))

# ── Run summary ───────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo " CTC scan complete  (${t_elapsed}s elapsed)"
echo "  ok      : ${#CASES_OK[@]}"
echo "  skipped : ${#CASES_SKIPPED[@]}"
echo "  error   : ${#CASES_ERROR[@]}"
[[ ${#CASES_SKIPPED[@]} -gt 0 ]] && echo "  skipped : ${CASES_SKIPPED[*]}"
[[ ${#CASES_ERROR[@]}   -gt 0 ]] && echo "  errors  : ${CASES_ERROR[*]}"
echo "=================================================="

# ── Barrier summary table ─────────────────────────────────────────────────────
if [[ ${#CASES_OK[@]} -eq 0 ]] && [[ "$MODE" == "single" ]]; then
    exit 0
fi

# Collect directories to summarise
summarise_dirs=()
if [[ "$MODE" == "single" ]]; then
    summarise_dirs=("$CASE_DIR_SINGLE")
else
    for cid in "${CASES_OK[@]}"; do
        summarise_dirs+=("${SWEEP_DIR}/${cid}")
    done
fi

[[ ${#summarise_dirs[@]} -eq 0 ]] && exit 0

"$PYTHON" - "${summarise_dirs[@]}" "${SUMMARY_CSV:-}" <<'PYEOF'
import json, csv, sys
from pathlib import Path

case_dirs = []
summary_csv_path = None
for arg in sys.argv[1:]:
    p = Path(arg)
    if p.is_dir():
        case_dirs.append(p)
    elif arg.endswith(".csv"):
        summary_csv_path = p

# Load existing sweep scores if available
scores = {}
if summary_csv_path and summary_csv_path.exists():
    try:
        with summary_csv_path.open() as f:
            for row in csv.DictReader(f):
                cid = row.get("case_id", "")
                try:
                    scores[cid] = float(row["score"])
                except (KeyError, TypeError, ValueError):
                    pass
    except Exception:
        pass

def find_report(d):
    for pat in ["*_sweep.json", "*_report.json", "report.json"]:
        hits = [p for p in d.glob(pat)
                if p.name != "params.json" and "_transport" not in p.name]
        if hits:
            return hits[0]
    jsons = [p for p in d.glob("*.json")
             if p.name != "params.json" and "_transport" not in p.name]
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

    ctc_barrier = r.get("transport_barrier_ctc_eV")
    xscan_barrier = r.get("transport_barrier_xscan_eV")
    if ctc_barrier is None:
        continue

    rows.append({
        "case_id":          cd.name,
        "ctc_barrier_eV":   float(ctc_barrier),
        "xscan_barrier_eV": float(xscan_barrier) if xscan_barrier is not None else None,
        "peak_x_um":        (r.get("transport_ctc_peak_x_m") or 0) * 1e6,
        "peak_y_um":        (r.get("transport_ctc_peak_y_m") or 0) * 1e6,
        "peak_z_um":        (r.get("transport_ctc_peak_z_m") or 0) * 1e6,
        "junction_x_um":    (r.get("transport_ctc_junction_x_m") or 0) * 1e6,
        "n_nodes":          r.get("transport_ctc_n_nodes", ""),
        "converged":        "yes" if r.get("transport_ctc_converged") else "no",
        "score":            scores.get(cd.name),
    })

rows.sort(key=lambda x: x["ctc_barrier_eV"])

print()
print(f"{'Rank':<5} {'case_id':<14} {'CTC meV':>10} {'xscan meV':>10} "
      f"{'peak_x µm':>10} {'peak_z µm':>10} {'cvg':>5} {'score':>8}")
print("─" * 78)
for i, row in enumerate(rows, 1):
    xscan_str = f"{row['xscan_barrier_eV']*1e3:.2f}" if row["xscan_barrier_eV"] is not None else "—"
    score_str = f"{row['score']:.4g}" if row["score"] is not None else "—"
    print(
        f"{i:<5} {row['case_id']:<14} "
        f"{row['ctc_barrier_eV']*1e3:>10.3f} "
        f"{xscan_str:>10} "
        f"{row['peak_x_um']:>10.1f} "
        f"{row['peak_z_um']:>10.1f} "
        f"{row['converged']:>5} "
        f"{score_str:>8}"
    )
print()

# Write CTC barriers CSV
if summary_csv_path is not None:
    out_csv = summary_csv_path.parent / "ctc_barriers.csv"
else:
    out_csv = Path(case_dirs[0].parent) / "ctc_barriers.csv" if case_dirs else None

if out_csv is not None and rows:
    fieldnames = ["case_id", "ctc_barrier_eV", "ctc_barrier_meV",
                  "xscan_barrier_eV", "xscan_barrier_meV",
                  "peak_x_um", "peak_y_um", "peak_z_um",
                  "junction_x_um", "n_nodes", "converged", "score"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out_row = dict(row)
            out_row["ctc_barrier_meV"]   = round(row["ctc_barrier_eV"] * 1e3, 4)
            out_row["xscan_barrier_meV"] = (
                round(row["xscan_barrier_eV"] * 1e3, 4)
                if row["xscan_barrier_eV"] is not None else None
            )
            w.writerow(out_row)
    print(f"CTC barrier results → {out_csv}")
PYEOF
