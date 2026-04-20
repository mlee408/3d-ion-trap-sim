#!/usr/bin/env python3
"""
make_mesh_4win_halft.py  –  Geometry + mesh generation for the 4-window half-thickness RF cell.

Usage (single case / smoke test):
    cd trap_sim/
    python scripts/make_mesh_4win_halft.py \
        --rf-thickness 1.0 \
        --out meshes/generated/rf_4win_halft_t100.msh

Usage (in automate.py sweep via sweeps/sweep_4win_halft.py):
    python scripts/make_mesh_4win_halft.py \
        --rf-thickness {rf_thickness} \
        --out {mesh_path}

Parameters
----------
window_count=4 is HARDCODED (window_n=2, giving a 2×2 symmetric arrangement).
--rf-height     H   Support-beam+lattice-top height [µm].  Default 145 µm = 50% of the
                    290 µm baseline.  Changing this varies the structural thickness.
--rf-thickness  T   Rib cross-section scale factor (default 1.0).
                    Window clear width = rib_spacing − dh_base × T
                                       = 300 − 56 × T  [µm].
                    At T=1.0: window width = 244 µm (well above fab minimum).
                    At T=0.6: window width = 266 µm.
                    At T=1.2: window width = 233 µm.
--lc-electrode      Near-electrode mesh size [mm] (default 0.003)
--lc-center         Trap-centre mesh size [mm]    (default 0.005)
--lc-far            Far-field mesh size [mm]       (default 0.035)
--pad-z-top         Vacuum padding above cell [mm] (default 0.500)
--out               Output .msh file path (required)

Thickness note
--------------
The baseline 3D RF lattice-cell height is rf_height = 290 µm (vertical extent
of the parametric structure above the rf_surface base plate).  This script
defaults to rf_height = 145 µm (exactly 50% of the 290 µm baseline), halving
the main 3D RF body dimension.  The rib cross-section scale factor rf_thickness
(affecting rib width, not body height) is left at 1.0 by default and may be
swept independently.

Assembly
--------
The 4-window RF cell (rf_cell_gen.py output) is combined with:
  • rf_surface.step  – RF base plate with X-shaped cutout (fused into RF cell)
  • dc.step          – DC electrodes  (physical group 2)
  • ground.step      – Ground planes  (physical group 3)
by assemble_mesh.py, which also creates the vacuum domain and mesh.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE        = Path(__file__).resolve().parent   # scripts/
ROOT        = HERE.parent                        # trap_sim/
RF_CELL_GEN = ROOT / "geometry" / "rf_cell_gen.py"
MESH_SCRIPT = ROOT / "geometry" / "assemble_mesh.py"
DC_STEP     = ROOT / "cad" / "base" / "dc.step"
GND_STEP    = ROOT / "cad" / "base" / "ground.step"
RF_SURFACE  = ROOT / "cad" / "base" / "rf_surface.step"

# --- hardcoded for the 4-window sample ---
WINDOW_COUNT             = 4      # total windows (symmetric 2×2 arrangement)
WINDOW_N                 = 2      # per side: WINDOW_N² = WINDOW_COUNT
RF_HEIGHT_BASELINE_UM    = 290.0  # original 3D RF body height
RF_HEIGHT_HALFT_DEFAULT  = RF_HEIGHT_BASELINE_UM / 2.0  # 145.0 µm

# Mesh quality defaults (shared with make_mesh_parametric.py)
OPT_LC_ELECTRODE = 0.003
OPT_LC_CENTER    = 0.005
OPT_LC_FAR       = 0.035
OPT_PAD_Z_TOP    = 0.500


def main() -> int:
    ap = argparse.ArgumentParser(
        description="4-window half-thickness RF cell: geometry + mesh pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--rf-height", type=float, default=RF_HEIGHT_HALFT_DEFAULT,
        help=(
            "Support-beam height [µm].  Default 145 = 50%% of 290 µm baseline.  "
            "This is the main vertical dimension of the 3D RF body being halved."
        ),
    )
    ap.add_argument(
        "--rf-thickness", type=float, default=1.0,
        help="Rib cross-section scale factor; window width = 300 − 56×T µm",
    )
    ap.add_argument("--lc-electrode", type=float, default=OPT_LC_ELECTRODE,
                    help="Near-electrode mesh size [mm]")
    ap.add_argument("--lc-center",    type=float, default=OPT_LC_CENTER,
                    help="Trap-centre mesh size [mm]")
    ap.add_argument("--lc-far",       type=float, default=OPT_LC_FAR,
                    help="Far-field mesh size [mm]")
    ap.add_argument("--pad-z-top",    type=float, default=OPT_PAD_Z_TOP,
                    help="Vacuum box padding above RF cell [mm]")
    ap.add_argument("--out", required=True, help="Output .msh file path")
    args = ap.parse_args()

    for p in (RF_CELL_GEN, MESH_SCRIPT, DC_STEP, GND_STEP, RF_SURFACE):
        if not p.exists():
            print(f"ERROR: required file not found: {p}", file=sys.stderr)
            return 1

    rf_height    = args.rf_height
    rf_thickness = args.rf_thickness

    out_path = Path(args.out).resolve()
    case_dir = out_path.parent
    case_dir.mkdir(parents=True, exist_ok=True)

    t_int     = int(round(rf_thickness * 100))
    step_path = case_dir / f"rfcell_h{int(rf_height)}_t{t_int:03d}_n{WINDOW_N}.step"

    print(
        f"\n[make_mesh_4win_halft]  4-window half-thickness RF cell\n"
        f"  window_n     = {WINDOW_N}  ({WINDOW_N}×{WINDOW_N} = {WINDOW_COUNT} windows)\n"
        f"  rf_height    = {rf_height} µm  "
        f"(baseline {RF_HEIGHT_BASELINE_UM} µm, ratio {rf_height/RF_HEIGHT_BASELINE_UM:.2f})\n"
        f"  rf_thickness = {rf_thickness}  "
        f"(window width ≈ {300 - 56*rf_thickness:.1f} µm clear gap)\n"
        f"  out          = {out_path}",
        flush=True,
    )

    # Step 1 — generate RF cell STEP (includes fusing rf_surface base plate)
    cmd_gen = [
        sys.executable, str(RF_CELL_GEN),
        "--window_n",     str(WINDOW_N),
        "--rf_height",    str(rf_height),
        "--rf_thickness", str(rf_thickness),
        "--base-step",    str(RF_SURFACE),
        "--step",
        "--no-brep",
    ]
    print(f"[make_mesh_4win_halft] GEN: {' '.join(cmd_gen)}", flush=True)
    rc = subprocess.run(cmd_gen, cwd=str(case_dir)).returncode
    if rc != 0:
        print(f"ERROR: rf_cell_gen.py exited {rc}", file=sys.stderr)
        return rc

    if not step_path.exists():
        print(f"ERROR: expected STEP at {step_path}", file=sys.stderr)
        return 1

    # Step 2 — full mesh pipeline: RF + DC + ground + vacuum + physical tags
    cmd_mesh = [
        sys.executable, str(MESH_SCRIPT),
        "--rf",           str(step_path),
        "--dc",           str(DC_STEP),
        "--ground",       str(GND_STEP),
        "--lc-electrode", str(args.lc_electrode),
        "--lc-center",    str(args.lc_center),
        "--lc-far",       str(args.lc_far),
        "--pad-z-top",    str(args.pad_z_top),
        "--nopopup",
        "--out", str(out_path),
    ]
    print(f"[make_mesh_4win_halft] MESH: {' '.join(cmd_mesh)}", flush=True)
    return subprocess.run(cmd_mesh).returncode


if __name__ == "__main__":
    raise SystemExit(main())
