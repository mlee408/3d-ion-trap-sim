#!/usr/bin/env python3
"""
make_rf_step.py — Generate complete electrode STEP files from the surface
junction generator, optionally fusing 3D RF lattice beams on top.

Pipeline
--------
1. Call ``generate_surface_junction.generate()`` to produce surface-layer
   electrode STEP files (rf.step, dc.step, ground.step) in ``--out-dir``.

2. (Optional, with ``--with-3d-rf``) Call ``rf_cell_gen.build_rf_cell()`` to
   generate a 3D RF lattice cell fused with the surface RF base plate,
   producing a single combined RF electrode STEP.

Usage
-----
Surface electrodes only (default)::

    python geometry/make_rf_step.py --out-dir cad/generated/surface_junction

With 3D RF beams::

    python geometry/make_rf_step.py --with-3d-rf --rf-height 267 --out-dir cad/generated/combined

Pass junction parameters through to the surface generator::

    python geometry/make_rf_step.py --o3-y-um 55 --o3-x-um 90 --out-dir cad/generated/narrow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE      = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

PAPER_RF_HEIGHT_UM = 267.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate electrode STEP files from surface junction generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Surface junction parameters (forwarded) ---
    ap.add_argument("--electrode-gap-um", type=float, default=10.0)
    ap.add_argument("--o1-x-um", type=float, default=5.0)
    ap.add_argument("--o1-y-um", type=float, default=5.0)
    ap.add_argument("--o1-r-um", type=float, default=0.0)
    ap.add_argument("--o2-x-um", type=float, default=23.07)
    ap.add_argument("--o2-y-um", type=float, default=5.0)
    ap.add_argument("--o2-r-um", type=float, default=0.0)
    ap.add_argument("--o3-x-um", type=float, default=95.27)
    ap.add_argument("--o3-y-um", type=float, default=60.0)
    ap.add_argument("--o3-r-um", type=float, default=0.0)
    ap.add_argument("--dc-min-width-um", type=float, default=20.0)
    ap.add_argument("--dc-segment-gap-um", type=float, default=10.0)
    ap.add_argument("--dc-junction-keepout-um", type=float, default=0.0)
    ap.add_argument("--ground-min-width-um", type=float, default=10.0)

    # --- 3D RF cell parameters ---
    ap.add_argument(
        "--with-3d-rf", action="store_true",
        help="Also generate 3D RF lattice beams fused with the surface RF.",
    )
    ap.add_argument(
        "--rf-height", type=float, default=PAPER_RF_HEIGHT_UM, metavar="UM",
        help="3D RF beam height [µm] (distance from surface to beam centroid).",
    )
    ap.add_argument(
        "--window-n", type=int, default=2, metavar="N",
        help="Number of lattice windows per side.",
    )

    # --- Output ---
    ap.add_argument(
        "--out-dir", type=str, default="cad/generated/surface_junction",
        help="Output directory for STEP/SVG/PNG files.",
    )
    ap.add_argument("--validate", action="store_true")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    # ── Step 1: Generate surface junction electrodes ──────────────────────────
    sys.path.insert(0, str(HERE))
    from generate_surface_junction import generate, DEFAULTS

    p = {
        "cell_size_um": 600.0,
        "electrode_gap_um": args.electrode_gap_um,
        "o1_x_um": args.o1_x_um,
        "o1_y_um": args.o1_y_um,
        "o1_r_um": args.o1_r_um,
        "o2_x_um": args.o2_x_um,
        "o2_y_um": args.o2_y_um,
        "o2_r_um": args.o2_r_um,
        "o3_x_um": args.o3_x_um,
        "o3_y_um": args.o3_y_um,
        "o3_r_um": args.o3_r_um,
        "dc_min_width_um": args.dc_min_width_um,
        "dc_segment_length_um": DEFAULTS["dc_segment_length_um"],
        "dc_segment_gap_um": args.dc_segment_gap_um,
        "dc_junction_keepout_um": args.dc_junction_keepout_um,
        "ground_min_width_um": args.ground_min_width_um,
        "out_dir": str(out_dir),
        "validate": args.validate,
    }

    result = generate(p)

    rf_step = out_dir / "rf.step"
    if not rf_step.exists():
        sys.exit("ERROR: surface junction generator did not produce rf.step")

    # ── Step 2 (optional): Fuse 3D RF lattice beams ──────────────────────────
    if args.with_3d_rf:
        from rf_cell_gen import build_rf_cell

        rf_height_um = args.rf_height
        print(
            f"\n{'='*60}\n"
            f"Fusing 3D RF lattice (rf_height={rf_height_um:.0f} µm, "
            f"window_n={args.window_n})\n"
            f"  base plate: {rf_step}\n"
            f"{'='*60}"
        )

        build_rf_cell(
            rf_height=rf_height_um,
            rf_thickness=1.0,
            window_n=args.window_n,
            out_brep=False,
            out_step=True,
            out_mesh=False,
            gui=False,
            base_step_path=str(rf_step),
        )

    # ── Verify output ─────────────────────────────────────────────────────────
    try:
        import cadquery as cq
        s = cq.importers.importStep(str(rf_step))
        b = s.val().BoundingBox()
        print(
            f"\n[verify] {rf_step.name}: "
            f"x=[{b.xmin:.4f},{b.xmax:.4f}]  "
            f"y=[{b.ymin:.4f},{b.ymax:.4f}]  "
            f"z=[{b.zmin:.4f},{b.zmax:.4f}] mm"
        )
    except (ImportError, Exception):
        pass

    # ── Next steps ────────────────────────────────────────────────────────────
    print(
        f"\nOutput directory: {out_dir}\n"
        f"  rf.step, dc.step, ground.step, layout.svg, layout.png\n\n"
        f"Next: generate mesh\n\n"
        f"  python geometry/assemble_mesh.py \\\n"
        f"      --rf     {out_dir}/rf.step \\\n"
        f"      --dc     {out_dir}/dc.step \\\n"
        f"      --ground {out_dir}/ground.step \\\n"
        f"      --njunctions 2 \\\n"
        f"      --out    meshes/generated/trap.msh \\\n"
        f"      --nopopup\n"
    )


if __name__ == "__main__":
    main()
