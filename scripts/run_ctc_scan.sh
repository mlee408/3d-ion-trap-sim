#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_ctc_scan.sh
#
# Run the paper-method CTC (Constant Total Confinement) path scan to compute
# the transport barrier for one or more sweep cases.  Uses --transport-mode
# both in compute_transport_barrier.py, which traces (a) the pseudopotential
# minimum path and (b) the CTC isosurface path through the junction.
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
#   # Use a fixed CTC target instead of auto-calibration:
#   CTC_TARGET_CONF=7.5e8 bash run_ctc_scan.sh
#
# Environment overrides
# ---------------------
#   PYTHON=/path/to/python3          (default: fenicsx conda env)
#   TRANSPORT_SCRIPT=/path/to/...    (default: ../src/compute_transport_barrier.py)
#   JUNCTION_PITCH=600e-6            (metres, default: 600 µm)
#   OVERWRITE=1                      (re-run even if barrier already present)
#   CTC_N_X=120                      (x-axis scan points, default: 120)
#   CTC_N_Z=300                      (z-axis scan points per x, default: 300)
#   CTC_Z_MIN=10e-6                  (z scan lower bound in metres, default: 10e-6)
#   CTC_Z_MAX=200e-6                 (z scan upper bound in metres, default: 200e-6)
#   CTC_HESSIAN_STEP_UM=2.0          (finite-diff step in µm, default: 2.0)
#   CTC_TARGET_CONF=                 (eV/m²; unset → --ctc-target-auto)
#   TRANSPORT_MODE=both              (both|min|ctc, default: both)
#   DEPTH_CORRECTION=1.3             (correction factor for mesh resolution vs COMSOL; default: 1.3)
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
JUNCTION_PITCH="${JUNCTION_PITCH:-600e-6}"
OVERWRITE_FLAG=""
[[ "${OVERWRITE:-0}" == "1" ]] && OVERWRITE_FLAG="--overwrite"
DC_VOLTAGES="${DC_VOLTAGES:-}"
DC_VOLTAGES_FLAG=""
[[ -n "$DC_VOLTAGES" ]] && DC_VOLTAGES_FLAG="--dc-voltages $DC_VOLTAGES"

TRANSPORT_MODE="${TRANSPORT_MODE:-both}"
CTC_N_X="${CTC_N_X:-120}"
CTC_N_Z="${CTC_N_Z:-300}"
CTC_Z_MIN="${CTC_Z_MIN:-10e-6}"
CTC_Z_MAX="${CTC_Z_MAX:-200e-6}"
CTC_HESSIAN_STEP_UM="${CTC_HESSIAN_STEP_UM:-2.0}"
CTC_TARGET_CONF="${CTC_TARGET_CONF:-0.75e9}"   # paper's fixed C = 0.75×10⁹ eV/m²
DEPTH_CORRECTION="${DEPTH_CORRECTION:-1.3}"

# Build target-conf flag
CTC_TARGET_FLAG="--ctc-target-auto"
if [[ -n "$CTC_TARGET_CONF" ]]; then
    CTC_TARGET_FLAG="--ctc-target-conf $CTC_TARGET_CONF"
fi

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
        --junction-pitch)
            JUNCTION_PITCH="$2"
            shift 2
            ;;
        --transport-mode)
            TRANSPORT_MODE="$2"
            shift 2
            ;;
        --ctc-n-x)
            CTC_N_X="$2"
            shift 2
            ;;
        --ctc-n-z)
            CTC_N_Z="$2"
            shift 2
            ;;
        --ctc-z-min)
            CTC_Z_MIN="$2"
            shift 2
            ;;
        --ctc-z-max)
            CTC_Z_MAX="$2"
            shift 2
            ;;
        --ctc-hessian-step-um)
            CTC_HESSIAN_STEP_UM="$2"
            shift 2
            ;;
        --ctc-target-conf)
            CTC_TARGET_FLAG="--ctc-target-conf $2"
            shift 2
            ;;
        --ctc-target-auto)
            CTC_TARGET_FLAG="--ctc-target-auto"
            shift
            ;;
        --dc-voltages)
            DC_VOLTAGES="$2"
            DC_VOLTAGES_FLAG="--dc-voltages $DC_VOLTAGES"
            shift 2
            ;;
        --help|-h)
            sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'
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
echo " CTC Transport Barrier Scan (paper method)"
echo " mode          : $MODE"
[[ -n "$SWEEP_DIR"        ]] && echo " sweep dir     : $SWEEP_DIR"
[[ -n "$CASE_DIR_SINGLE"  ]] && echo " case dir      : $CASE_DIR_SINGLE"
[[ ${#CASE_NUMS[@]} -gt 0 ]] && echo " case numbers  : ${CASE_NUMS[*]}"
echo " transport mode: $TRANSPORT_MODE"
echo " ctc target    : $CTC_TARGET_FLAG"
echo " ctc n_x       : $CTC_N_X"
echo " ctc n_z       : $CTC_N_Z"
echo " ctc z range   : [$CTC_Z_MIN, $CTC_Z_MAX] m"
echo " hessian step  : $CTC_HESSIAN_STEP_UM µm"
echo " junction pitch: $JUNCTION_PITCH m"
echo " dc_voltages   : ${DC_VOLTAGES:-(none — RF-only barrier)}"
echo " depth corr    : ${DEPTH_CORRECTION}"
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
    # shellcheck disable=SC2086
    "$PYTHON" "$TRANSPORT_SCRIPT" \
        --case-dir "$case_dir" \
        --transport-mode "$TRANSPORT_MODE" \
        --junction-pitch "$JUNCTION_PITCH" \
        --output-mode patch \
        $CTC_TARGET_FLAG \
        --ctc-n-x "$CTC_N_X" \
        --ctc-n-z "$CTC_N_Z" \
        --ctc-z-min "$CTC_Z_MIN" \
        --ctc-z-max "$CTC_Z_MAX" \
        --ctc-hessian-step-um "$CTC_HESSIAN_STEP_UM" \
        --depth-correction "$DEPTH_CORRECTION" \
        $OVERWRITE_FLAG \
        $DC_VOLTAGES_FLAG \
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

    # Paper-method keys (new)
    ctc_barrier   = r.get("ctc_barrier_eV")
    min_barrier   = r.get("transport_min_barrier_eV")
    ctc_target    = r.get("ctc_target_conf_eV_per_m2")
    max_dz        = r.get("ctc_max_abs_dz_um")
    rms_dz        = r.get("ctc_rms_dz_um")
    no_cross      = r.get("ctc_no_crossing_count", 0)

    # Fall back to legacy string-method key if paper keys absent
    if ctc_barrier is None:
        ctc_barrier = r.get("transport_barrier_ctc_eV")

    if ctc_barrier is None:
        continue

    rows.append({
        "case_id":          cd.name,
        "ctc_barrier_meV":  float(ctc_barrier) * 1e3,
        "min_barrier_meV":  float(min_barrier) * 1e3 if min_barrier is not None else None,
        "ctc_target_eV_m2": ctc_target,
        "max_dz_um":        max_dz,
        "rms_dz_um":        rms_dz,
        "no_crossing":      no_cross,
        "score":            scores.get(cd.name),
    })

rows.sort(key=lambda x: x["ctc_barrier_meV"])

print()
print(f"{'Rank':<5} {'case_id':<14} {'CTC meV':>10} {'min meV':>10} "
      f"{'max|dz|µm':>10} {'rms_dz µm':>10} {'no_cross':>9} {'score':>8}")
print("─" * 82)
for i, row in enumerate(rows, 1):
    min_str   = f"{row['min_barrier_meV']:.3f}"  if row["min_barrier_meV"] is not None else "—"
    dz_str    = f"{row['max_dz_um']:.2f}"        if row["max_dz_um"]       is not None else "—"
    rms_str   = f"{row['rms_dz_um']:.2f}"        if row["rms_dz_um"]       is not None else "—"
    score_str = f"{row['score']:.4g}"             if row["score"]           is not None else "—"
    print(
        f"{i:<5} {row['case_id']:<14} "
        f"{row['ctc_barrier_meV']:>10.3f} "
        f"{min_str:>10} "
        f"{dz_str:>10} "
        f"{rms_str:>10} "
        f"{row['no_crossing']:>9} "
        f"{score_str:>8}"
    )
print()

# Write CTC barriers CSV
if summary_csv_path is not None:
    out_csv = summary_csv_path.parent / "ctc_barriers.csv"
else:
    out_csv = Path(case_dirs[0].parent) / "ctc_barriers.csv" if case_dirs else None

if out_csv is not None and rows:
    fieldnames = ["case_id",
                  "ctc_barrier_eV", "ctc_barrier_meV",
                  "min_barrier_eV", "min_barrier_meV",
                  "ctc_target_eV_m2",
                  "max_dz_um", "rms_dz_um", "no_crossing",
                  "score"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out_row = dict(row)
            out_row["ctc_barrier_eV"] = row["ctc_barrier_meV"] / 1e3
            out_row["min_barrier_eV"] = (
                row["min_barrier_meV"] / 1e3
                if row["min_barrier_meV"] is not None else None
            )
            w.writerow(out_row)
    print(f"CTC barrier results → {out_csv}")
PYEOF
