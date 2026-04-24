#!/usr/bin/env bash
# wait_and_run_top_barriers.sh
#
# Polls a sweep directory until all expected cases have completed (or errored),
# then runs compute_transport_barrier.py on the top-N scoring cases.
#
# Usage (run from repo root):
#   bash scripts/wait_and_run_top_barriers.sh \
#       runs/sweeps/n4_thickness_height_refined_yb2025 \
#       [--top N]   (default 8)
#       [--poll 60] (seconds between checks, default 60)
#
# The script reads automation_config.json for the expected case count, then
# polls summary.csv until that many rows have a non-empty status field.
# It then calls scripts/run_top_barriers.sh with the top-N case indices sorted
# by score (descending).

set -euo pipefail

SWEEP_DIR="${1:?Usage: $0 <sweep_dir> [--top N] [--poll S]}"
shift

TOP_N=8
POLL_S=60

while [[ $# -gt 0 ]]; do
    case "$1" in
        --top)  TOP_N="$2";  shift 2 ;;
        --poll) POLL_S="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

SWEEP_DIR="$(realpath "$SWEEP_DIR")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMMARY_CSV="${SWEEP_DIR}/summary.csv"
AUTOMATION_CFG="${SWEEP_DIR}/automation_config.json"
RUN_BARRIERS="${SCRIPT_DIR}/run_top_barriers.sh"

# ── Validate paths ────────────────────────────────────────────────────────────
[[ -d "$SWEEP_DIR"        ]] || { echo "ERROR: sweep_dir not found: $SWEEP_DIR" >&2; exit 1; }
[[ -f "$AUTOMATION_CFG"   ]] || { echo "ERROR: automation_config.json not found" >&2; exit 1; }
[[ -f "$RUN_BARRIERS"     ]] || { echo "ERROR: run_top_barriers.sh not found at: $RUN_BARRIERS" >&2; exit 1; }

# ── Read expected case count ──────────────────────────────────────────────────
N_CASES=$(python3 -c "import json; print(json.load(open('${AUTOMATION_CFG}'))['n_cases'])")
echo "=================================================="
echo " wait_and_run_top_barriers"
echo " sweep dir   : $SWEEP_DIR"
echo " expected    : $N_CASES cases"
echo " top-N       : $TOP_N"
echo " poll every  : ${POLL_S}s"
echo " barriers via: $RUN_BARRIERS"
echo "=================================================="
echo ""

# ── Poll until all cases have a status ───────────────────────────────────────
count_done() {
    # Count rows in summary.csv with a non-empty status (exclude header, skip blanks)
    [[ -f "$SUMMARY_CSV" ]] || { echo 0; return; }
    python3 - <<PYEOF
import csv
from pathlib import Path
p = Path("${SUMMARY_CSV}")
if not p.exists():
    print(0); exit()
done = 0
with p.open() as f:
    for row in csv.DictReader(f):
        if row.get("status", "").strip():
            done += 1
print(done)
PYEOF
}

while true; do
    done=$(count_done)
    ts=$(date '+%H:%M:%S')
    printf "[%s] %d / %d cases complete\n" "$ts" "$done" "$N_CASES"
    if [[ "$done" -ge "$N_CASES" ]]; then
        echo ""
        echo "All $N_CASES cases complete — proceeding."
        break
    fi
    sleep "$POLL_S"
done

# ── Extract top-N case indices by score (descending) ─────────────────────────
TOP_CASE_NUMS=$(python3 - <<PYEOF
import csv
from pathlib import Path

p = Path("${SUMMARY_CSV}")
rows = []
with p.open() as f:
    for row in csv.DictReader(f):
        cid    = row.get("case_id", "")
        status = row.get("status", "").strip()
        score  = row.get("score", "").strip()
        if status != "ok" or not score:
            continue
        try:
            num = int(cid.split("_")[-1])
            rows.append((float(score), num, cid))
        except (ValueError, IndexError):
            pass

rows.sort(key=lambda x: x[0], reverse=True)
top = rows[:${TOP_N}]

print("Top ${TOP_N} by score:")
for rank, (sc, num, cid) in enumerate(top, 1):
    print(f"  #{rank:>2}  {cid}  score={sc:.4f}")
print()

# Emit case indices on a tagged line for the shell to parse from stdout
nums = [str(num) for _, num, _ in top]
print("CASE_NUMS:" + " ".join(nums))
PYEOF
)

# Split the tagged line from the display output
CASE_NUMS_LINE=$(echo "$TOP_CASE_NUMS" | grep '^CASE_NUMS:')
CASE_NUMS="${CASE_NUMS_LINE#CASE_NUMS:}"
echo "$TOP_CASE_NUMS" | grep -v '^CASE_NUMS:'

echo "Running barriers for cases: $CASE_NUMS"
echo ""

# ── Run barriers ──────────────────────────────────────────────────────────────
cd "$SWEEP_DIR"
# shellcheck disable=SC2086
bash "$RUN_BARRIERS" $CASE_NUMS
