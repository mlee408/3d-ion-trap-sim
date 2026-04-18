#!/usr/bin/env python3
"""
make_mesh.py  — automate.py wrapper for the parametric geometry sweep.

Place this file in src/ alongside automate.py and parametric_trap.py.

The mesh quality defaults below are the optimised values found by the
mesh-quality sweep (sweep_002/003).  Adjust them if your sweep found
better values.

Geometric parameters exposed to automate.py:
    {n_rails}       — rail grid density (1=1 opening, 2=4, 3=9, 4=16)
    {rail_width}    — rail bar width in µm
    {pillar_height} — RF pillar height in µm  (optional, default 288)

Example --mesh-template for automate.py (run from src/):

    python $(pwd)/make_mesh.py \\
        --n-rails {n_rails} \\
        --rail-width {rail_width} \\
        --out {mesh_path}

Or with pillar height also swept:

    python $(pwd)/make_mesh.py \\
        --n-rails {n_rails} \\
        --rail-width {rail_width} \\
        --pillar-height {pillar_height} \\
        --out {mesh_path}
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE            = Path(__file__).resolve().parent   # src/
PARAMETRIC_SCRIPT = HERE / "parametric_trap.py"

# ── Optimised mesh quality settings from automate.py sweep ───────────────
# Replace these with your best case from sweep_002/003 summary.csv.
# The values below are solid defaults if the sweep is still running.
OPT_LC_ELECTRODE  = 0.0028357770287737814   # mm — element size on electrode surfaces
OPT_LC_CENTER     = 0.01   # mm — element size at trap centre
OPT_LC_FAR        = 0.0343277049357128   # mm — far-field element size
OPT_PAD_Z_TOP     = 0.300   # mm — vacuum box height above rail tops


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Wrapper: calls parametric_trap.py with optimised mesh settings."
    )
    # Geometric params (swept by automate.py)
    ap.add_argument("--n-rails",       type=float, default=2.0,
                    help="Rail grid N (float, rounded internally). 1→1 opening, 2→4, 3→9.")
    ap.add_argument("--rail-width",    type=float, default=28.0,
                    help="Rail bar width in µm.")
    ap.add_argument("--pillar-height", type=float, default=288.0,
                    help="RF pillar height in µm.")

    # Mesh quality overrides (normally use baked-in optimised defaults)
    ap.add_argument("--lc-electrode",  type=float, default=OPT_LC_ELECTRODE)
    ap.add_argument("--lc-center",     type=float, default=OPT_LC_CENTER)
    ap.add_argument("--lc-far",        type=float, default=OPT_LC_FAR)
    ap.add_argument("--pad-z-top",     type=float, default=OPT_PAD_Z_TOP)

    ap.add_argument("--out", required=True, help="Output .msh path.")
    args = ap.parse_args()

    if not PARAMETRIC_SCRIPT.exists():
        print(f"ERROR: parametric_trap.py not found at {PARAMETRIC_SCRIPT}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable, str(PARAMETRIC_SCRIPT),
        "--n-rails",       str(args.n_rails),
        "--rail-width",    str(args.rail_width),
        "--pillar-height", str(args.pillar_height),
        "--lc-electrode",  str(args.lc_electrode),
        "--lc-center",     str(args.lc_center),
        "--lc-far",        str(args.lc_far),
        "--pad-z-top",     str(args.pad_z_top),
        "--nopopup",
        "--out", args.out,
    ]

    print(f"[make_mesh] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())