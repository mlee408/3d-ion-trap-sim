#!/usr/bin/env python3
"""
make_mesh_parametric.py

Wrapper for the geometry sweep:
  1. rf_cell_gen.py   -> generates parametric RF cell STEP
  2. assemble_mesh.py -> tiles N junctions + builds vacuum box + meshes

Place in scripts/ (or wherever your project keeps sweep scripts).
rf_cell_gen.py and assemble_mesh.py must be in geometry/.

Parameters swept by automate.py:
    {rf_width_um}  -- electrode width in um (min 10, step 5)
                      controls BOTH rail and pillar thickness equally
    {rf_height}    -- vertical distance from surface RF electrode top to
                      the centroid of the 3D RF beam cross-section [µm]
                      (NOT total beam height; NOT bottom-of-beam to substrate)
    {njunctions}   -- number of junctions (1/2/3/4, default 2)

window_n is fixed per sweep run (passed as $N in the bash for-loop).

Update OPT_* with best values from your mesh quality sweep.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE            = Path(__file__).resolve().parent   # scripts/
ROOT            = HERE.parent                        # trap_sim/
RF_CELL_GEN     = ROOT / "geometry" / "rf_cell_gen.py"
ASSEMBLE        = ROOT / "geometry" / "assemble_mesh.py"
DC_STEP         = ROOT / "cad" / "base" / "dc.step"
GND_STEP        = ROOT / "cad" / "base" / "ground.step"
RF_SURFACE_STEP = ROOT / "cad" / "base" / "rf_surface.step"

# Optimised mesh quality settings from sweep_mesh -- update these
OPT_LC_ELECTRODE = 0.003
OPT_LC_CENTER    = 0.005
OPT_LC_FAR       = 0.035
OPT_PAD_Z_TOP    = 0.600


def main() -> int:
    ap = argparse.ArgumentParser()
    # Geometric params swept by automate.py
    ap.add_argument("--window-n",    type=float, default=2.0,
                    help="Windows per side (rounded to int: 1/2/3/4)")
    ap.add_argument("--rf-width-um", type=float, default=56.0,
                    help="Electrode width in um (min 10, step 5). "
                         "Controls both rail and pillar thickness equally.")
    ap.add_argument("--rf-height",   type=float, default=290.0,
                    help="Vertical distance from surface RF electrode top to "
                         "RF beam cross-section centroid [µm]. "
                         "Beam spans ±(dv/2) around this centre in z.")
    ap.add_argument("--njunctions",  type=int, default=2,
                    help="Number of junctions to assemble (1/2/3/4). Default: 2")

    # Mesh quality
    ap.add_argument("--lc-electrode",  type=float, default=OPT_LC_ELECTRODE)
    ap.add_argument("--lc-center",     type=float, default=OPT_LC_CENTER)
    ap.add_argument("--lc-far",        type=float, default=OPT_LC_FAR)
    ap.add_argument("--pad-z-top",     type=float, default=OPT_PAD_Z_TOP)

    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for p in (RF_CELL_GEN, ASSEMBLE, DC_STEP, GND_STEP, RF_SURFACE_STEP):
        if not p.exists():
            print(f"ERROR: required file not found: {p}", file=sys.stderr)
            return 1

    window_n  = max(1, round(args.window_n))
    rf_width  = args.rf_width_um
    rf_height = args.rf_height
    njunctions = max(1, min(4, args.njunctions))

    out_path = Path(args.out).resolve()
    case_dir = out_path.parent

    print(f"[make_mesh] window_n={window_n}  rf_width={rf_width}um  "
          f"rf_height={rf_height}um (beam-centre above surface RF top)  "
          f"njunctions={njunctions}", flush=True)

    # ── Step 1: generate parametric RF cell STEP via rf_cell_gen.py ──────
    # rf_cell_gen.py writes to cwd, so run it from case_dir.
    # We do NOT use --njunctions here -- tiling is handled by assemble_mesh.py.
    cmd_gen = [
        sys.executable, str(RF_CELL_GEN),
        "--window_n",    str(window_n),
        "--rf_height",   str(rf_height),
        "--rf_width_um", str(rf_width),
        "--njunctions",  "1",          # generate ONE cell only;
                                       # assemble_mesh.py tiles to 2 junctions
        "--step",
        "--no-brep",
    ]
    print(f"[make_mesh] generating RF cell: {' '.join(cmd_gen)}", flush=True)
    rc = subprocess.run(cmd_gen, cwd=str(case_dir)).returncode
    if rc != 0:
        print(f"ERROR: rf_cell_gen.py failed (exit {rc})", file=sys.stderr)
        return rc

    # Find the STEP file rf_cell_gen.py wrote
    matches = sorted(case_dir.glob("rfcell_*.step"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        print(f"ERROR: no rfcell_*.step found in {case_dir}", file=sys.stderr)
        return 1
    rf_step = matches[0]
    print(f"[make_mesh] RF STEP: {rf_step.name}", flush=True)

    # ── Step 1b: merge rf_surface.step into the RF cell STEP ─────────────
    # Delegates to scripts/merge_rf_with_base.py so diagnostic output is clean
    # and failures surface with tracebacks instead of being swallowed.
    rf_with_base = case_dir / (rf_step.stem + "_wb.step")
    merge_script_path = HERE / "merge_rf_with_base.py"
    if not merge_script_path.exists():
        print(f"ERROR: missing {merge_script_path}", file=sys.stderr)
        return 1

    cmd_merge = [
        sys.executable, str(merge_script_path),
        str(rf_step),
        str(RF_SURFACE_STEP),
        str(rf_with_base),
    ]
    print(f"[make_mesh] merging: {' '.join(cmd_merge)}", flush=True)
    rc = subprocess.run(cmd_merge).returncode
    if rc != 0:
        print(f"ERROR: merge_rf_with_base.py failed (exit {rc})", file=sys.stderr)
        return rc
    print(f"[make_mesh] RF+surface STEP: {rf_with_base.name}", flush=True)

    # ── Step 2: mesh with assemble_mesh.py (handles N-junction tiling) ───
    # Pass the merged RF+surface STEP — assemble_mesh.py tiles it N times,
    # so all junctions get the base plate.
    cmd_mesh = [
        sys.executable, str(ASSEMBLE),
        "--rf",             str(rf_with_base),
        "--dc",             str(DC_STEP),
        "--ground",         str(GND_STEP),
        "--njunctions",     str(njunctions),
        "--junction-pitch", "0.600",    # 600 um centre-to-centre
        "--lc-electrode",   str(args.lc_electrode),
        "--lc-center",      str(args.lc_center),
        "--lc-far",         str(args.lc_far),
        "--pad-z-top",      str(args.pad_z_top),
        "--nopopup",
        "--out", str(out_path),
    ]
    print(f"[make_mesh] meshing: {' '.join(cmd_mesh)}", flush=True)
    return subprocess.run(cmd_mesh).returncode


if __name__ == "__main__":
    raise SystemExit(main())