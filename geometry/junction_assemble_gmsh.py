#!/usr/bin/env python3
"""
Assemble two identical X-junction STEP models using gmsh OCC.

Default inputs (in the same directory as this script):
    rf.step, dc.step, ground.step

Behavior:
- imports rf.step, dc.step, ground.step
- duplicates each imported volume set
- translates the duplicate by --spacing mm center-to-center (default 0.6 mm)
- performs OCC fragment/fuse-style cleanup so overlaps are accounted for
- groups resulting volumes by source class (RF/DC/GROUND)
- exports the combined geometry to connected_xjunction.step (or --out)
- optionally also writes .brep for debugging

Run (standalone):
    python junction_assemble_gmsh.py

Run with explicit paths (for use in automate.py mesh templates):
    python junction_assemble_gmsh.py \\
        --rf path/to/rf.step \\
        --dc path/to/dc.step \\
        --ground path/to/ground.step \\
        --out {case_dir}/combined.step \\
        --spacing 0.6

Requirements:
    pip install gmsh
or a local gmsh Python install.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import math

try:
    import gmsh
except Exception as e:
    raise SystemExit(
        "gmsh Python module is required. Install with e.g. 'pip install gmsh'.\n"
        f"Import error: {e}"
    )

CENTER_TO_CENTER_MM = 0.6   # 600 um
TRANSLATION_AXIS = "x"      # "x", "y", or "z"
WRITE_BREP = True
VERBOSE = True


def log(msg: str) -> None:
    if VERBOSE:
        print(msg)


def axis_vector(distance_mm: float, axis: str) -> tuple[float, float, float]:
    axis = axis.lower()
    if axis == "x":
        return (distance_mm, 0.0, 0.0)
    if axis == "y":
        return (0.0, distance_mm, 0.0)
    if axis == "z":
        return (0.0, 0.0, distance_mm)
    raise ValueError("TRANSLATION_AXIS must be one of: x, y, z")


def import_step_volumes(path: Path) -> list[tuple[int, int]]:
    before = set(gmsh.model.occ.getEntities(3))
    gmsh.model.occ.importShapes(str(path))
    after = set(gmsh.model.occ.getEntities(3))
    new_vols = sorted(after - before)
    if not new_vols:
        raise RuntimeError(f"No volumes imported from {path}")
    return new_vols


def copy_translate(dimtags: list[tuple[int, int]], dx: float, dy: float, dz: float) -> list[tuple[int, int]]:
    copied = gmsh.model.occ.copy(dimtags)
    gmsh.model.occ.translate(copied, dx, dy, dz)
    return copied


def bbox_of(dimtags: list[tuple[int, int]]) -> tuple[float, float, float, float, float, float]:
    xs0, ys0, zs0 = [], [], []
    xs1, ys1, zs1 = [], [], []
    for dim, tag in dimtags:
        x0, y0, z0, x1, y1, z1 = gmsh.model.occ.getBoundingBox(dim, tag)
        xs0.append(x0); ys0.append(y0); zs0.append(z0)
        xs1.append(x1); ys1.append(y1); zs1.append(z1)
    return (min(xs0), min(ys0), min(zs0), max(xs1), max(ys1), max(zs1))


def volume_com(dimtags: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[float, float, float]]]:
    out = []
    for dim, tag in dimtags:
        try:
            c = gmsh.model.occ.getCenterOfMass(dim, tag)
        except Exception:
            c = (math.nan, math.nan, math.nan)
        out.append(((dim, tag), c))
    return out


def point_in_bbox(pt: tuple[float, float, float], bb: tuple[float, float, float, float, float, float], tol=1e-9) -> bool:
    x, y, z = pt
    x0, y0, z0, x1, y1, z1 = bb
    return (x0 - tol <= x <= x1 + tol and y0 - tol <= y <= y1 + tol and z0 - tol <= z <= z1 + tol)


def classify_by_seed(all_vols: list[tuple[int, int]],
                     rf_seed_boxes,
                     dc_seed_boxes,
                     gnd_seed_boxes):
    rf, dc, gnd, unknown = [], [], [], []
    for dim, tag in all_vols:
        c = gmsh.model.occ.getCenterOfMass(dim, tag)
        if any(point_in_bbox(c, bb, tol=1e-6) for bb in rf_seed_boxes):
            rf.append((dim, tag))
        elif any(point_in_bbox(c, bb, tol=1e-6) for bb in dc_seed_boxes):
            dc.append((dim, tag))
        elif any(point_in_bbox(c, bb, tol=1e-6) for bb in gnd_seed_boxes):
            gnd.append((dim, tag))
        else:
            unknown.append((dim, tag))
    return rf, dc, gnd, unknown


def build_argparser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Assemble two identical X-junction STEP models using gmsh OCC."
    )
    ap.add_argument("--rf", type=Path, default=here / "rf.step",
                    help="RF electrode STEP file (default: rf.step in script dir)")
    ap.add_argument("--dc", type=Path, default=here / "dc.step",
                    help="DC electrode STEP file (default: dc.step in script dir)")
    ap.add_argument("--ground", type=Path, default=here / "ground.step",
                    help="Ground electrode STEP file (default: ground.step in script dir)")
    ap.add_argument("--out", type=Path, default=here / "connected_xjunction.step",
                    help="Output combined STEP file (default: connected_xjunction.step)")
    ap.add_argument("--spacing", type=float, default=CENTER_TO_CENTER_MM,
                    help=f"Junction center-to-center spacing in mm (default: {CENTER_TO_CENTER_MM})")
    ap.add_argument("--axis", type=str, default=TRANSLATION_AXIS, choices=["x", "y", "z"],
                    help=f"Translation axis (default: {TRANSLATION_AXIS})")
    ap.add_argument("--no-brep", action="store_true",
                    help="Skip writing .brep companion file")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress progress output")
    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()

    rf_path = args.rf
    dc_path = args.dc
    gnd_path = args.ground
    out_step = args.out
    out_brep = out_step.with_suffix(".brep")
    write_brep = not args.no_brep

    global VERBOSE
    VERBOSE = not args.quiet

    for p in (rf_path, dc_path, gnd_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    dx, dy, dz = axis_vector(args.spacing, args.axis)

    # Pass only program name to gmsh.initialize so argparse owns the rest
    gmsh.initialize([sys.argv[0]])
    gmsh.option.setNumber("General.Terminal", 1 if VERBOSE else 0)
    gmsh.model.add("connected_xjunction")

    try:
        log("Importing original STEP bodies...")
        rf1 = import_step_volumes(rf_path)
        dc1 = import_step_volumes(dc_path)
        gnd1 = import_step_volumes(gnd_path)
        gmsh.model.occ.synchronize()

        rf_seed_boxes = [bbox_of([v]) for v in rf1]
        dc_seed_boxes = [bbox_of([v]) for v in dc1]
        gnd_seed_boxes = [bbox_of([v]) for v in gnd1]

        log(f"RF volumes imported    : {len(rf1)}")
        log(f"DC volumes imported    : {len(dc1)}")
        log(f"GROUND volumes imported: {len(gnd1)}")

        log(f"Creating translated copy by ({dx}, {dy}, {dz}) mm...")
        rf2 = copy_translate(rf1, dx, dy, dz)
        dc2 = copy_translate(dc1, dx, dy, dz)
        gnd2 = copy_translate(gnd1, dx, dy, dz)
        gmsh.model.occ.synchronize()

        rf2_seed_boxes = [bbox_of([v]) for v in rf2]
        dc2_seed_boxes = [bbox_of([v]) for v in dc2]
        gnd2_seed_boxes = [bbox_of([v]) for v in gnd2]

        all_seed_rf = rf_seed_boxes + rf2_seed_boxes
        all_seed_dc = dc_seed_boxes + dc2_seed_boxes
        all_seed_gnd = gnd_seed_boxes + gnd2_seed_boxes

        all_in = rf1 + rf2 + dc1 + dc2 + gnd1 + gnd2

        log("Fragmenting all imported volumes to account for overlaps...")
        # Fragment is more robust than fuse when many touching/overlapping solids exist.
        gmsh.model.occ.fragment(all_in, [])
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        all_vols = sorted(gmsh.model.occ.getEntities(3))
        log(f"Total resulting volumes after fragment/removeAllDuplicates: {len(all_vols)}")

        rf_vols, dc_vols, gnd_vols, unknown = classify_by_seed(
            all_vols, all_seed_rf, all_seed_dc, all_seed_gnd
        )

        # If some fragments fall outside seed bounding boxes due to splitting/boolean quirks,
        # keep them visible and report them.
        if unknown:
            log(f"Warning: {len(unknown)} volume(s) could not be confidently classified.")
            for dim, tag in unknown[:10]:
                log(f"  unknown volume tag={tag}, COM={gmsh.model.occ.getCenterOfMass(dim, tag)}")

        if rf_vols:
            gmsh.model.addPhysicalGroup(3, [t for _, t in rf_vols], tag=1, name="RF")
        if dc_vols:
            gmsh.model.addPhysicalGroup(3, [t for _, t in dc_vols], tag=2, name="DC")
        if gnd_vols:
            gmsh.model.addPhysicalGroup(3, [t for _, t in gnd_vols], tag=3, name="GROUND")
        if unknown:
            gmsh.model.addPhysicalGroup(3, [t for _, t in unknown], tag=99, name="UNKNOWN")

        gmsh.model.occ.synchronize()

        log(f"RF classified volumes    : {len(rf_vols)}")
        log(f"DC classified volumes    : {len(dc_vols)}")
        log(f"GROUND classified volumes: {len(gnd_vols)}")

        if write_brep:
            gmsh.write(str(out_brep))
            log(f"Wrote: {out_brep}")

        gmsh.write(str(out_step))
        log(f"Wrote: {out_step}")

        return 0
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
