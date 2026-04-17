#!/usr/bin/env python3
"""
make_mesh.py  — thin wrapper around meshes/run_case.py for use with automate.py.

Place this file in src/ (next to automate.py).  automate.py calls it via
--mesh-template with the four mesh-quality parameters substituted in.

Usage (handled automatically by automate.py):
    python make_mesh.py \
        --lc-electrode 0.003 \
        --lc-center 0.005 \
        --lc-far 0.030 \
        --pad-z-top 0.600 \
        --out /path/to/case_XXXX/mesh.msh

The STEP file paths are resolved relative to this script's location,
so this file must stay in src/ alongside automate.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # src/
REPO_ROOT = HERE.parent                          # 3d-ion-trap-sim/
MESH_SCRIPT = REPO_ROOT / "meshes" / "run_case.py"
STEP_DIR = REPO_ROOT / "meshes" / "step"

RF_STEP     = STEP_DIR / "rf.step"
DC_STEP     = STEP_DIR / "dc.step"
GROUND_STEP = STEP_DIR / "ground.step"


def main() -> int:
    ap = argparse.ArgumentParser(description="Mesh-quality wrapper for automate.py")
    ap.add_argument("--lc-electrode", type=float, default=0.003,
                    help="Element size on electrode surfaces (mm)")
    ap.add_argument("--lc-center",   type=float, default=0.005,
                    help="Element size at trap centre ball (mm)")
    ap.add_argument("--lc-far",      type=float, default=0.030,
                    help="Element size in far field (mm)")
    ap.add_argument("--pad-z-top",   type=float, default=0.600,
                    help="Vacuum box height above electrodes (mm)")
    ap.add_argument("--out", required=True,
                    help="Output .msh path (supplied by automate.py as {mesh_path})")
    args = ap.parse_args()

    for p in (MESH_SCRIPT, RF_STEP, DC_STEP, GROUND_STEP):
        if not p.exists():
            print(f"ERROR: required file not found: {p}", file=sys.stderr)
            return 1

    cmd = [
        sys.executable, str(MESH_SCRIPT),
        "--rf",     str(RF_STEP),
        "--dc",     str(DC_STEP),
        "--ground", str(GROUND_STEP),
        "--lc-electrode", str(args.lc_electrode),
        "--lc-center",    str(args.lc_center),
        "--lc-far",       str(args.lc_far),
        "--pad-z-top",    str(args.pad_z_top),
        "--nopopup",
        "--out", args.out,
    ]

    print(f"[make_mesh] running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())