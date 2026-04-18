#!/usr/bin/env python3
"""
make_mesh_parametric.py

Wrapper for the geometry sweep using rf_cell_gen.py (partner's parametric
RF cell generator) + geometry/assemble_mesh.py (mesh pipeline).

Place this file in scripts/.

Parameters swept by automate.py:
    {window_n}      -- windows per side (1/2/3/4 -> 1/4/9/16 openings)
    {rf_height}     -- support beam height in um (default 290)
    {rf_thickness}  -- lattice beam cross-section scale factor (default 1.0)

Mesh quality defaults: update OPT_* with best values from sweep_mesh.
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
rf_surface_STEP = ROOT / "cad" / "base" / "rf_surface.step" 

# Optimised mesh quality settings from sweep_mesh -- update these
OPT_LC_ELECTRODE = 0.003
OPT_LC_CENTER    = 0.005
OPT_LC_FAR       = 0.035
OPT_PAD_Z_TOP    = 0.500

def main() -> int:
    ap = argparse.ArgumentParser()
    # Geometric params swept by automate.py
    ap.add_argument("--window-n",    type=float, default=2.0,
                    help="Windows per side (rounded to int: 1/2/3/4)")
    ap.add_argument("--rf-width-um", type=float, default=56.0,
                    help="Electrode width in µm (min 10, step 5). "
                         "Controls both rail and pillar thickness equally.")
    ap.add_argument("--rf-height",   type=float, default=290.0,
                    help="Support beam height in µm")
 
    # Mesh quality
    ap.add_argument("--lc-electrode",  type=float, default=OPT_LC_ELECTRODE)
    ap.add_argument("--lc-center",     type=float, default=OPT_LC_CENTER)
    ap.add_argument("--lc-far",        type=float, default=OPT_LC_FAR)
    ap.add_argument("--pad-z-top",     type=float, default=OPT_PAD_Z_TOP)
 
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
 
    for p in (RF_CELL_GEN, MESH_SCRIPT, DC_STEP, GND_STEP, rf_surface_STEP):
        if not p.exists():
            print(f"ERROR: required file not found: {p}", file=sys.stderr)
            return 1
 
    window_n   = max(1, round(args.window_n))
    rf_width   = args.rf_width_um
    rf_height  = args.rf_height
 
    out_path  = Path(args.out).resolve()
    case_dir  = out_path.parent
    t_int     = int(round(rf_width))
    step_path = case_dir / f"rfcell_h{int(rf_height)}_w{t_int}_n{window_n}.step"
 
    print(f"[make_mesh] window_n={window_n}  rf_width={rf_width}um  "
          f"rf_height={rf_height}um", flush=True)
 
    # Step 1: generate RF cell STEP via rf_cell_gen.py
    cmd_gen = [
        sys.executable, str(RF_CELL_GEN),
        "--window_n",   str(window_n),
        "--rf_height",  str(rf_height),
        "--rf_width_um", str(rf_width),   # width in µm — same for rails and pillars
        "--base-step",  str(rf_surface_STEP),
        "--step",
        "--no-brep",
    ]
    print(f"[make_mesh] {' '.join(cmd_gen)}", flush=True)
    rc = subprocess.run(cmd_gen, cwd=str(case_dir)).returncode
    if rc != 0:
        print(f"ERROR: rf_cell_gen.py failed (exit {rc})", file=sys.stderr)
        return rc
 
    # rf_cell_gen writes the file using its own stem — find it
    # stem format: rfcell_h{height}_t{thickness*100:03d}_n{n}.step
    # With rf_width_um the stem uses rf_thickness = rf_width_um / 56.0
    import glob
    matches = list(case_dir.glob("rfcell_*.step"))
    if not matches:
        print(f"ERROR: no rfcell_*.step found in {case_dir}", file=sys.stderr)
        return 1
    # Use the most recently created one
    rf_step = max(matches, key=lambda p: p.stat().st_mtime)
    print(f"[make_mesh] using RF STEP: {rf_step.name}", flush=True)
 
    # Step 2: mesh pipeline
    cmd_mesh = [
        sys.executable, str(MESH_SCRIPT),
        "--rf",     str(rf_step),
        "--dc",     str(DC_STEP),
        "--ground", str(GND_STEP),
        "--lc-electrode", str(args.lc_electrode),
        "--lc-center",    str(args.lc_center),
        "--lc-far",       str(args.lc_far),
        "--pad-z-top",    str(args.pad_z_top),
        "--nopopup",
        "--out", str(out_path),
    ]
    print(f"[make_mesh] {' '.join(cmd_mesh)}", flush=True)
    return subprocess.run(cmd_mesh).returncode
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 