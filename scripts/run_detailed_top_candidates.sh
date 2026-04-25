#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_detailed_top_candidates.sh
#
# Full-detail metric computation for the top candidate from each parameter
# sweep.  Three passes per case:
#   1. Linear region  – r0 searched between the two junction centers
#   2. Junction region – r0 searched near junction center (0, 0)
#   3. Transport barrier – pseudopotential saddle scan junction→junction
#
# Geometry: 2 junction cells, each 600 µm × 600 µm.
#   Junction 0 center: (0, 0)      mesh coords
#   Junction 1 center: (0.6, 0)    mesh coords
#   Linear region:      x ∈ [~0.15, ~0.45]
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."          # project root

# ── Shared physics parameters (Paper 2: Yb-171, 44.3 MHz, 190 V) ────────────
COMMON=(
  --degree 2
  --rf-freq  44300000.0
  --mass-amu 171.0
  --charge-e 1.0
  --vrf      190.0
  --coord-unit 0.001
  --rf-tags  1
  --ground-tags 2 3
  --outer-tags  4
  --save-solution
)

# ── Top candidates (one per sweep) ──────────────────────────────────────────
CASES="
n4_case0035 runs/sweeps/n4_thickness_height_sweep_yb2025/case_0035
n4_case0024 runs/sweeps/n4_thickness_height_sweep_yb2025/case_0024
n4_case0014 runs/sweeps/n4_thickness_height_sweep_yb2025/case_0014
n4_case0007 runs/sweeps/n4_thickness_height_sweep_yb2025/case_0007
n4_case0020 runs/sweeps/n4_thickness_height_sweep_yb2025/case_0020
"

# ── Run function ─────────────────────────────────────────────────────────────
run_metrics() {
  local label="$1" mesh="$2" outdir="$3" prefix="$4"
  shift 4
  # remaining args are r0 bounds
  echo ""
  echo "================================================================"
  echo "  $label"
  echo "  mesh:   $mesh"
  echo "  outdir: $outdir"
  echo "================================================================"
  python src/run_sweep_metrics.py \
    --mesh "$mesh" \
    --outdir "$outdir" \
    --prefix "$prefix" \
    "${COMMON[@]}" \
    "$@" \
    2>&1 | tee "$outdir/${prefix}_stdout.txt"
}

run_transport() {
  local label="$1" casedir="$2"
  echo ""
  echo "================================================================"
  echo "  TRANSPORT BARRIER: $label"
  echo "================================================================"
  python src/compute_transport_barrier.py \
    --case-dir "$casedir" \
    --junction-pitch 600e-6 \
    --n-steps 120 \
    --scan-both-axes \
    --overwrite \
    --output-mode sidecar \
    2>&1 | tee "${casedir}/transport_detailed_stdout.txt"
}

# ── Main loop ────────────────────────────────────────────────────────────────
echo "$CASES" | while read -r tag CASEDIR; do
  [ -z "${tag:-}" ] && continue
  MESH="$CASEDIR/mesh.msh"

  if [ ! -f "$MESH" ]; then
    echo "SKIP $tag — mesh not found: $MESH"
    continue
  fi

  # ── 1. Linear region (r0 between junctions) ─────────────────────────────
  OUTDIR_LIN="$CASEDIR/detailed_linear"
  mkdir -p "$OUTDIR_LIN"
  run_metrics \
    "LINEAR  $tag" "$MESH" "$OUTDIR_LIN" "${tag}_linear" \
    --r0-x-min 0.12  --r0-x-max 0.48 \
    --r0-y-min -0.18 --r0-y-max 0.18 \
    --r0-z-min 0.02  --r0-z-max 0.15

  # ── 2. Junction region (r0 near junction center at origin) ──────────────
  OUTDIR_JCT="$CASEDIR/detailed_junction"
  mkdir -p "$OUTDIR_JCT"
  run_metrics \
    "JUNCTION  $tag" "$MESH" "$OUTDIR_JCT" "${tag}_junction" \
    --r0-x-min -0.18 --r0-x-max 0.18 \
    --r0-y-min -0.18 --r0-y-max 0.18 \
    --r0-z-min 0.01  --r0-z-max 0.12

  # ── 3. Transport barrier (junction-to-junction scan) ────────────────────
  # Uses the linear-region report for r0 starting point
  # Copy linear report to case dir so compute_transport_barrier finds it
  LIN_REPORT=$(ls "$OUTDIR_LIN"/*_sweep.json 2>/dev/null | head -1)
  if [ -n "$LIN_REPORT" ]; then
    cp "$LIN_REPORT" "$CASEDIR/${tag}_detailed_sweep.json"
    # Also copy checkpoint if saved
    LIN_CKPT=$(ls "$OUTDIR_LIN"/*_phi_rf_dofs.npy 2>/dev/null | head -1)
    if [ -n "$LIN_CKPT" ]; then
      cp "$LIN_CKPT" "$CASEDIR/${tag}_detailed_phi_rf_dofs.npy"
    fi
  fi
  run_transport "$tag" "$CASEDIR"

done

echo ""
echo "========================================"
echo "  ALL DONE"
echo "========================================"
echo ""

# ── Summary table ────────────────────────────────────────────────────────────
echo "SUMMARY — Linear Region"
echo "────────────────────────────────────────────────────────────────"
printf "%-16s %8s %8s %10s %10s %10s\n" \
       "case" "z+ (eV)" "z- (eV)" "radial" "freq_min" "freq_max"
echo "────────────────────────────────────────────────────────────────"
for tag in "${!CASES[@]}"; do
  OUTDIR="${CASES[$tag]}/detailed_linear"
  REP=$(ls "$OUTDIR"/*_sweep.json 2>/dev/null | head -1)
  [ -z "$REP" ] && continue
  python3 -c "
import json, sys
d = json.load(open('$REP'))
print(f\"${tag:<16s} {d.get('depth_z_plus_eV',0):8.3f} {d.get('depth_z_minus_eV',0):8.3f} \
{d.get('radial_depth_core_eV',0):10.4f} {d.get('strong_freq_min_hz',0)/1e6:10.4f} \
{d.get('strong_freq_max_hz',0)/1e6:10.4f}\")
" 2>/dev/null
done

echo ""
echo "SUMMARY — Junction Region"
echo "────────────────────────────────────────────────────────────────"
printf "%-16s %8s %8s %10s %10s %10s\n" \
       "case" "z+ (eV)" "z- (eV)" "radial" "freq_min" "freq_max"
echo "────────────────────────────────────────────────────────────────"
for tag in "${!CASES[@]}"; do
  OUTDIR="${CASES[$tag]}/detailed_junction"
  REP=$(ls "$OUTDIR"/*_sweep.json 2>/dev/null | head -1)
  [ -z "$REP" ] && continue
  python3 -c "
import json
d = json.load(open('$REP'))
print(f\"${tag:<16s} {d.get('depth_z_plus_eV',0):8.3f} {d.get('depth_z_minus_eV',0):8.3f} \
{d.get('radial_depth_core_eV',0):10.4f} {d.get('strong_freq_min_hz',0)/1e6:10.4f} \
{d.get('strong_freq_max_hz',0)/1e6:10.4f}\")
" 2>/dev/null
done
