#!/usr/bin/env python3
"""
make_rf_step.py — Generate complete electrode STEP files for surface junction
trap with optional 3D RF lattice beams.

Outputs
-------
<out-dir>/
  rf_surface.step          single-cell surface RF electrode
  dc.step                  single-cell DC electrodes
  ground.step              single-cell ground electrodes
  surface_combined.step    all surface electrodes (single cell)
  rf_3d.step               3D RF lattice beams only (single cell)
  layout.svg / layout.png  visual layout
  params.json              junction parameters
  1j/
    rf.step                surface RF + 3D RF fused (1 junction)
    dc.step                DC electrodes
    ground.step            ground electrodes
    combined.step          all electrodes
  2j/
    rf.step                tiled surface RF + 3D RF fused (2 junctions)
    dc.step                tiled DC electrodes
    ground.step            tiled ground electrodes
    combined.step          all electrodes

Usage
-----
::

    python geometry/make_rf_step.py --rf-height 247 --out-dir cad/generated/baseline
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

HERE      = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

PAPER_RF_HEIGHT_UM = 247.0
JUNCTION_PITCH_MM  = 0.6       # 600 µm


# ---------------------------------------------------------------------------
# OCC helpers for STEP manipulation
# ---------------------------------------------------------------------------

def _import_occ():
    from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.gp import gp_Trsf, gp_Vec
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    from OCP.Interface import Interface_Static
    return {
        "STEPControl_Reader": STEPControl_Reader,
        "STEPControl_Writer": STEPControl_Writer,
        "STEPControl_AsIs": STEPControl_AsIs,
        "BRepBuilderAPI_Transform": BRepBuilderAPI_Transform,
        "BRepAlgoAPI_Fuse": BRepAlgoAPI_Fuse,
        "gp_Trsf": gp_Trsf,
        "gp_Vec": gp_Vec,
        "TopoDS_Compound": TopoDS_Compound,
        "BRep_Builder": BRep_Builder,
        "Interface_Static": Interface_Static,
    }


def read_step(path, occ):
    reader = occ["STEPControl_Reader"]()
    reader.ReadFile(str(path))
    reader.TransferRoots()
    return reader.OneShape()


def write_step(shape, path, occ):
    writer = occ["STEPControl_Writer"]()
    occ["Interface_Static"].SetCVal_s("write.step.schema", "AP203")
    writer.Transfer(shape, occ["STEPControl_AsIs"])
    if writer.Write(str(path)) != 1:
        raise RuntimeError(f"STEP write failed: {path}")


def translate_shape(shape, dx_mm, dy_mm, dz_mm, occ):
    trsf = occ["gp_Trsf"]()
    trsf.SetTranslation(occ["gp_Vec"](dx_mm, dy_mm, dz_mm))
    return occ["BRepBuilderAPI_Transform"](shape, trsf, True).Shape()


def make_compound(shapes, occ):
    builder = occ["BRep_Builder"]()
    compound = occ["TopoDS_Compound"]()
    builder.MakeCompound(compound)
    for s in shapes:
        builder.Add(compound, s)
    return compound


def fuse_shapes(shapes, occ):
    if len(shapes) == 0:
        raise ValueError("No shapes to fuse")
    if len(shapes) == 1:
        return shapes[0]
    result = shapes[0]
    for s in shapes[1:]:
        result = occ["BRepAlgoAPI_Fuse"](result, s).Shape()
    return result


def tile_compound(step_path, n, pitch_mm, occ):
    """Tile a STEP file n times along x, return as compound (separate bodies)."""
    shape = read_step(step_path, occ)
    copies = [shape]
    for i in range(1, n):
        copies.append(translate_shape(shape, pitch_mm * i, 0, 0, occ))
    return make_compound(copies, occ)


def tile_fused(step_path, n, pitch_mm, occ):
    """Tile a STEP file n times along x, fuse into single solid."""
    shape = read_step(step_path, occ)
    copies = [shape]
    for i in range(1, n):
        copies.append(translate_shape(shape, pitch_mm * i, 0, 0, occ))
    return fuse_shapes(copies, occ)


# ---------------------------------------------------------------------------
# 3D RF generation via rf_cell_gen
# ---------------------------------------------------------------------------

def generate_3d_rf(rf_height_um, window_n, njunctions, out_dir):
    """Call build_rf_cell and return the path to the output STEP file."""
    from rf_cell_gen import build_rf_cell

    prev_cwd = os.getcwd()
    os.chdir(str(out_dir))
    try:
        build_rf_cell(
            rf_height=rf_height_um,
            rf_thickness=1.0,
            window_n=window_n,
            njunctions=njunctions,
            out_brep=False,
            out_step=True,
            out_mesh=False,
            gui=False,
            base_step_path=None,
        )
    finally:
        os.chdir(prev_cwd)

    t_int = int(round(1.0 * 100))
    stem = f"rfcell_h{int(rf_height_um)}_t{t_int:03d}_n{window_n}_j{njunctions}"
    return out_dir / f"{stem}.step"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_step(path, label=""):
    try:
        import cadquery as cq
        s = cq.importers.importStep(str(path))
        b = s.val().BoundingBox()
        n_solids = len(s.solids().vals())
        print(
            f"  [verify] {label or path.name}: "
            f"x=[{b.xmin:.4f},{b.xmax:.4f}]  "
            f"y=[{b.ymin:.4f},{b.ymax:.4f}]  "
            f"z=[{b.zmin:.4f},{b.zmax:.4f}] mm  "
            f"({n_solids} solid{'s' if n_solids != 1 else ''})"
        )
    except (ImportError, Exception):
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate complete electrode STEP files for surface junction trap.",
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
        "--rf-height", type=float, default=PAPER_RF_HEIGHT_UM, metavar="UM",
        help="3D RF beam height [µm] (distance from surface to beam centroid).",
    )
    ap.add_argument(
        "--window-n", type=int, default=2, metavar="N",
        help="Number of lattice windows per side.",
    )

    # --- Output ---
    ap.add_argument(
        "--out-dir", type=str, default="cad/generated/baseline",
        help="Output directory for all generated files.",
    )
    ap.add_argument("--validate", action="store_true")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: Generate single-cell surface electrodes
    # ══════════════════════════════════════════════════════════════════════════
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

    generate(p)

    # Rename surface outputs to final names
    (out_dir / "rf.step").rename(out_dir / "rf_surface.step")
    (out_dir / "combined.step").rename(out_dir / "surface_combined.step")
    print(f"\n  rf.step → rf_surface.step")
    print(f"  combined.step → surface_combined.step")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: Generate 3D RF beams (single junction, no surface base)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Generating 3D RF lattice (1 junction, rf_height={args.rf_height:.0f} µm)")
    print(f"{'='*60}")

    rf_3d_1j_path = generate_3d_rf(args.rf_height, args.window_n, 1, out_dir)
    rf_3d_1j_path.rename(out_dir / "rf_3d.step")
    print(f"  {rf_3d_1j_path.name} → rf_3d.step")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: Assemble 1-junction cell
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Assembling 1-junction cell")
    print(f"{'='*60}")

    occ = _import_occ()
    dir_1j = out_dir / "1j"
    dir_1j.mkdir(parents=True, exist_ok=True)

    rf_surface = read_step(out_dir / "rf_surface.step", occ)
    rf_3d = read_step(out_dir / "rf_3d.step", occ)

    rf_1j = fuse_shapes([rf_surface, rf_3d], occ)
    write_step(rf_1j, dir_1j / "rf.step", occ)
    print(f"  Wrote 1j/rf.step (surface + 3D fused)")

    shutil.copy2(out_dir / "dc.step", dir_1j / "dc.step")
    shutil.copy2(out_dir / "ground.step", dir_1j / "ground.step")
    print(f"  Copied dc.step, ground.step → 1j/")

    dc_1j = read_step(dir_1j / "dc.step", occ)
    gnd_1j = read_step(dir_1j / "ground.step", occ)
    combined_1j = make_compound([rf_1j, dc_1j, gnd_1j], occ)
    write_step(combined_1j, dir_1j / "combined.step", occ)
    print(f"  Wrote 1j/combined.step")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4: Assemble 2-junction cell
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Assembling 2-junction cell")
    print(f"{'='*60}")

    dir_2j = out_dir / "2j"
    dir_2j.mkdir(parents=True, exist_ok=True)

    # Tile surface RF (fuse into single solid so rails connect)
    rf_surface_2j = tile_fused(out_dir / "rf_surface.step", 2, JUNCTION_PITCH_MM, occ)

    # Generate 3D RF beams for 2 junctions
    print(f"\n  Generating 3D RF lattice (2 junctions)...")
    rf_3d_2j_path = generate_3d_rf(args.rf_height, args.window_n, 2, dir_2j)

    rf_3d_2j = read_step(rf_3d_2j_path, occ)
    rf_3d_2j_path.unlink()

    # Fuse tiled surface RF with 2-junction 3D beams
    rf_2j = fuse_shapes([rf_surface_2j, rf_3d_2j], occ)
    write_step(rf_2j, dir_2j / "rf.step", occ)
    print(f"  Wrote 2j/rf.step (tiled surface + 3D fused)")

    # Tile DC and ground (compound — separate bodies)
    dc_2j = tile_compound(out_dir / "dc.step", 2, JUNCTION_PITCH_MM, occ)
    write_step(dc_2j, dir_2j / "dc.step", occ)
    print(f"  Wrote 2j/dc.step")

    gnd_2j = tile_compound(out_dir / "ground.step", 2, JUNCTION_PITCH_MM, occ)
    write_step(gnd_2j, dir_2j / "ground.step", occ)
    print(f"  Wrote 2j/ground.step")

    combined_2j = make_compound([rf_2j, dc_2j, gnd_2j], occ)
    write_step(combined_2j, dir_2j / "combined.step", occ)
    print(f"  Wrote 2j/combined.step")

    # ══════════════════════════════════════════════════════════════════════════
    # Verify outputs
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Verification")
    print(f"{'='*60}")
    verify_step(out_dir / "rf_surface.step", "rf_surface.step")
    verify_step(out_dir / "rf_3d.step", "rf_3d.step")
    verify_step(dir_1j / "rf.step", "1j/rf.step")
    verify_step(dir_2j / "rf.step", "2j/rf.step")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")
    print(f"  rf_surface.step        surface RF (single cell)")
    print(f"  dc.step                DC electrodes (single cell)")
    print(f"  ground.step            ground electrodes (single cell)")
    print(f"  surface_combined.step  all surface electrodes (single cell)")
    print(f"  rf_3d.step             3D RF lattice beams (single cell)")
    print(f"  1j/rf.step             complete RF (1 junction)")
    print(f"  1j/combined.step       all electrodes (1 junction)")
    print(f"  2j/rf.step             complete RF (2 junctions)")
    print(f"  2j/combined.step       all electrodes (2 junctions)")


if __name__ == "__main__":
    main()
