#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_base_paper_test.sh
#
# Full-detail metric computation for the original paper geometry (cad/base/rf.step).
# Three passes:
#   1. Linear region  – r0 searched between the two junction centers
#   2. Junction region – r0 searched near junction center (0, 0)
#   3. Transport barrier – pseudopotential saddle scan junction→junction
#
# Geometry: 2 junction cells, each 600 µm × 600 µm, pitch 0.6 mm.
#   Junction 0 center: (0, 0)    mesh coords
#   Junction 1 center: (0.6, 0)  mesh coords
#   Linear region:      x ∈ [~0.12, ~0.48]
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."          # project root

CASEDIR="runs/base_paper_test"
MESH="$CASEDIR/mesh.msh"
TAG="base_paper"

# ── Shared physics parameters (Paper: Yb-171, 44.3 MHz, 190 V) ──────────────
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

# ── Step 1: Generate mesh if not already present ─────────────────────────────
if [ -f "$MESH" ]; then
  echo "[mesh] Already exists: $MESH — skipping mesh generation."
else
  echo "[mesh] Generating 2-junction mesh from cad/base STEP files..."
  mkdir -p "$CASEDIR/mesh_xdmf"
  python geometry/assemble_mesh.py \
    --rf     cad/base/rf.step \
    --dc     cad/base/dc.step \
    --ground cad/base/ground.step \
    --njunctions 2 \
    --junction-pitch 0.6 \
    --lc-electrode 0.003 \
    --lc-center    0.005 \
    --lc-far       0.035 \
    --pad-z-top    0.600 \
    --nopopup \
    --out "$MESH" \
    2>&1 | tee "$CASEDIR/meshgen.stdout.txt"
  echo "[mesh] Done."
fi

# ── Run helpers ───────────────────────────────────────────────────────────────
run_metrics() {
  local label="$1" outdir="$2" prefix="$3"
  shift 3
  echo ""
  echo "================================================================"
  echo "  $label"
  echo "  mesh:   $MESH"
  echo "  outdir: $outdir"
  echo "================================================================"
  python src/run_sweep_metrics.py \
    --mesh "$MESH" \
    --outdir "$outdir" \
    --prefix "$prefix" \
    "${COMMON[@]}" \
    "$@" \
    2>&1 | tee "$outdir/${prefix}_stdout.txt"
}

run_transport() {
  echo ""
  echo "================================================================"
  echo "  TRANSPORT BARRIER: $TAG"
  echo "================================================================"
  python src/compute_transport_barrier.py \
    --case-dir "$CASEDIR" \
    --junction-pitch 600e-6 \
    --n-steps 120 \
    --scan-both-axes \
    --overwrite \
    --output-mode sidecar \
    2>&1 | tee "$CASEDIR/transport_detailed_stdout.txt"
}

# ── Pass 1: Linear region ────────────────────────────────────────────────────
OUTDIR_LIN="$CASEDIR/detailed_linear"
mkdir -p "$OUTDIR_LIN"
run_metrics \
  "LINEAR  $TAG" "$OUTDIR_LIN" "${TAG}_linear" \
  --r0-x-min 0.12  --r0-x-max 0.48 \
  --r0-y-min -0.18 --r0-y-max 0.18 \
  --r0-z-min 0.02  --r0-z-max 0.15

# ── Pass 2: Junction region ──────────────────────────────────────────────────
OUTDIR_JCT="$CASEDIR/detailed_junction"
mkdir -p "$OUTDIR_JCT"
run_metrics \
  "JUNCTION  $TAG" "$OUTDIR_JCT" "${TAG}_junction" \
  --r0-x-min -0.18 --r0-x-max 0.18 \
  --r0-y-min -0.18 --r0-y-max 0.18 \
  --r0-z-min 0.01  --r0-z-max 0.12

# ── Pass 3: Transport barrier ────────────────────────────────────────────────
# Copy linear report to case dir so compute_transport_barrier finds it
LIN_REPORT=$(ls "$OUTDIR_LIN"/*_sweep.json 2>/dev/null | head -1)
if [ -n "$LIN_REPORT" ]; then
  cp "$LIN_REPORT" "$CASEDIR/${TAG}_detailed_sweep.json"
  LIN_CKPT=$(ls "$OUTDIR_LIN"/*_phi_rf_dofs.npy 2>/dev/null | head -1)
  if [ -n "$LIN_CKPT" ]; then
    cp "$LIN_CKPT" "$CASEDIR/${TAG}_detailed_phi_rf_dofs.npy"
  fi
fi
run_transport

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  ALL DONE — base_paper_test"
echo "========================================"

for region in linear junction; do
  OUTDIR="$CASEDIR/detailed_${region}"
  REP=$(ls "$OUTDIR"/*_sweep.json 2>/dev/null | head -1)
  [ -z "$REP" ] && continue
  echo ""
  echo "SUMMARY — ${region^^} region"
  echo "────────────────────────────────────────────────────────────────"
  python3 -c "
import json
d = json.load(open('$REP'))
print(f\"r0 = ({d.get('r0_x_m',0)*1e6:.1f}, {d.get('r0_y_m',0)*1e6:.1f}, {d.get('r0_z_m',0)*1e6:.1f}) µm\")
print(f\"freq1/2/3  = {d.get('freq1_hz',0)/1e6:.4f} / {d.get('freq2_hz',0)/1e6:.4f} / {d.get('freq3_hz',0)/1e6:.4f} MHz\")
print(f\"depth_z    = {d.get('depth_z_eV',0):.4f} eV  (z+ {d.get('depth_z_plus_eV',0):.4f}  z- {d.get('depth_z_minus_eV',0):.4f})\")
print(f\"depth_y    = {d.get('depth_y_eV',0):.4f} eV\")
print(f\"radial     = {d.get('radial_depth_core_eV',0):.4f} eV\")
print(f\"transport  = {d.get('transport_barrier_xscan_eV', 'N/A')} eV (x-scan)\")
" 2>/dev/null
done
