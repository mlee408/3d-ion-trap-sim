#!/usr/bin/env python3
"""RF-first surface-electrode X-junction generator.

RF geometry is the primary design variable. DC electrodes are derived
automatically from the RF layout via a fixed 10 µm RF–DC clearance gap.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
from shapely.ops import unary_union

from OCP.gp import gp_Pnt, gp_Vec
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
)
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.TopoDS import TopoDS_Shape, TopoDS_Compound
from OCP.BRep import BRep_Builder
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.Interface import Interface_Static
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

# ---------------------------------------------------------------------------
# Fixed constraints
# ---------------------------------------------------------------------------
CELL_HALF_UM = 300.0
CELL_SIZE_UM = 600.0
GAP_UM = 10.0
THICKNESS_UM = 10.0

Z_BOT = -THICKNESS_UM
Z_TOP = 0.0


def _mm(x: float) -> float:
    return x * 1e-3


# ---------------------------------------------------------------------------
# OCC helpers
# ---------------------------------------------------------------------------

def _extrude_polygon(verts_um: list[tuple[float, float]]) -> TopoDS_Shape:
    n = len(verts_um)
    wire_builder = BRepBuilderAPI_MakeWire()
    for i in range(n):
        x0, y0 = verts_um[i]
        x1, y1 = verts_um[(i + 1) % n]
        edge = BRepBuilderAPI_MakeEdge(
            gp_Pnt(_mm(x0), _mm(y0), _mm(Z_BOT)),
            gp_Pnt(_mm(x1), _mm(y1), _mm(Z_BOT)),
        ).Edge()
        wire_builder.Add(edge)
    face = BRepBuilderAPI_MakeFace(wire_builder.Wire(), True).Face()
    h = _mm(Z_TOP - Z_BOT)
    return BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, h)).Shape()


def _make_compound(shapes: list[TopoDS_Shape]) -> TopoDS_Shape:
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for s in shapes:
        builder.Add(compound, s)
    return compound


def _write_step(shape: TopoDS_Shape, path: str) -> None:
    writer = STEPControl_Writer()
    Interface_Static.SetCVal_s("write.step.schema", "AP203")
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(path)
    if status != 1:
        raise RuntimeError(f"STEP write failed with status {status}: {path}")


def _shape_bbox(shape: TopoDS_Shape) -> tuple[float, float, float, float, float, float]:
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    x0, y0, z0, x1, y1, z1 = box.Get()
    return x0 * 1000, y0 * 1000, z0 * 1000, x1 * 1000, y1 * 1000, z1 * 1000


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(p: dict) -> None:
    rw = p["rf_rail_width_um"]
    cg = p["center_gap_um"]
    ji = p["junction_inner_um"]
    jo = p["junction_outer_um"]
    cr = p["junction_curve_radius_um"]
    gi = cg / 2.0
    go = gi + rw

    errors: list[str] = []
    if rw <= 0:
        errors.append("rf_rail_width_um must be > 0")
    if cg <= 2 * GAP_UM:
        errors.append(f"center_gap_um must be > {2 * GAP_UM}")
    if ji <= 0:
        errors.append("junction_inner_um must be > 0")
    if ji >= jo:
        errors.append("junction_inner_um must be < junction_outer_um")
    if jo > gi:
        errors.append("junction_outer_um must be <= center_gap_um / 2")
    if go >= CELL_HALF_UM:
        errors.append(f"go = {go:.1f} must be < cell_half_um = {CELL_HALF_UM}")
    if cr < 0:
        errors.append("junction_curve_radius_um must be >= 0")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# RF 2D geometry
# ---------------------------------------------------------------------------

def _rf_polygons_2d(p: dict) -> list[list[tuple[float, float]]]:
    gi = p["center_gap_um"] / 2.0
    go = gi + p["rf_rail_width_um"]
    ji = p["junction_inner_um"]
    jo = p["junction_outer_um"]
    C = CELL_HALF_UM

    polys: list[list[tuple[float, float]]] = []
    for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
        # L-shaped rail
        polys.append([
            (sx * gi, sy * gi),
            (sx * C,  sy * gi),
            (sx * C,  sy * go),
            (sx * go, sy * go),
            (sx * go, sy * C),
            (sx * gi, sy * C),
        ])
        # Junction cap triangle
        polys.append([
            (sx * ji, sy * ji),
            (sx * ji, sy * jo),
            (sx * jo, sy * ji),
        ])

    return polys


def _build_rf_shapely(p: dict) -> ShapelyPolygon:
    raw = _rf_polygons_2d(p)
    shapely_polys = [ShapelyPolygon(v) for v in raw]
    union = unary_union(shapely_polys)

    cr = p["junction_curve_radius_um"]
    if cr > 0:
        union = union.buffer(cr, join_style="round", resolution=64)
        union = union.buffer(-cr, join_style="round", resolution=64)

    return union


# ---------------------------------------------------------------------------
# DC 2D geometry (derived from RF)
# ---------------------------------------------------------------------------

def _dc_candidates(p: dict) -> list[list[tuple[float, float]]]:
    gi = p["center_gap_um"] / 2.0
    go = gi + p["rf_rail_width_um"]
    C = CELL_HALF_UM

    dc_arm_width = gi
    dc_arm_outer = go + GAP_UM + 30.0
    if dc_arm_outer > C:
        dc_arm_outer = C

    candidates: list[list[tuple[float, float]]] = []

    # Center control strips along each arm (+x, -x, +y, -y)
    # +x arm
    candidates.append([
        (gi, -dc_arm_width),
        (C,  -dc_arm_width),
        (C,   dc_arm_width),
        (gi,  dc_arm_width),
    ])
    # -x arm
    candidates.append([
        (-C,  -dc_arm_width),
        (-gi, -dc_arm_width),
        (-gi,  dc_arm_width),
        (-C,   dc_arm_width),
    ])
    # +y arm
    candidates.append([
        (-dc_arm_width, gi),
        ( dc_arm_width, gi),
        ( dc_arm_width, C),
        (-dc_arm_width, C),
    ])
    # -y arm
    candidates.append([
        (-dc_arm_width, -C),
        ( dc_arm_width, -C),
        ( dc_arm_width, -gi),
        (-dc_arm_width, -gi),
    ])

    # Junction center control pad
    candidates.append([
        (-gi, -gi),
        ( gi, -gi),
        ( gi,  gi),
        (-gi,  gi),
    ])

    # Outer flank pads (one per quadrant, outside RF rails)
    for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
        candidates.append([
            (sx * go, sy * go),
            (sx * C,  sy * go),
            (sx * C,  sy * C),
            (sx * go, sy * C),
        ])

    return candidates


def _collect_poly(poly: ShapelyPolygon, out: list[list[tuple[float, float]]]) -> None:
    if poly.is_empty or poly.area < 1.0:
        return
    coords = list(poly.exterior.coords)
    if coords[-1] == coords[0]:
        coords = coords[:-1]
    if len(coords) >= 3:
        out.append(coords)


def _clip_dc(rf_union: ShapelyPolygon, p: dict) -> list[list[tuple[float, float]]]:
    rf_keepout = rf_union.buffer(GAP_UM, join_style="mitre", mitre_limit=10.0)
    candidates = _dc_candidates(p)
    clipped: list[list[tuple[float, float]]] = []

    for verts in candidates:
        dc_shape = ShapelyPolygon(verts)
        result = dc_shape.difference(rf_keepout)
        if result.is_empty:
            continue
        if isinstance(result, ShapelyPolygon):
            _collect_poly(result, clipped)
        elif isinstance(result, MultiPolygon):
            for poly in result.geoms:
                _collect_poly(poly, clipped)

    return clipped


# ---------------------------------------------------------------------------
# SVG debug output
# ---------------------------------------------------------------------------

def _write_svg(rf_polys_2d: list[list[tuple[float, float]]],
               dc_polys: list[list[tuple[float, float]]],
               rf_union: ShapelyPolygon,
               path: str) -> None:
    C = CELL_HALF_UM
    margin = 40
    size = 2 * (C + margin)
    ox = C + margin
    oy = C + margin

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
        f'<rect width="{size}" height="{size}" fill="#f8f8f8"/>',
        f'<line x1="{ox}" y1="0" x2="{ox}" y2="{size}" stroke="#ddd" stroke-width="0.5"/>',
        f'<line x1="0" y1="{oy}" x2="{size}" y2="{oy}" stroke="#ddd" stroke-width="0.5"/>',
    ]

    def _svg_poly(verts, fill, stroke="#333", sw=0.5, opacity=0.8):
        pts = " ".join(f"{ox + v[0]:.2f},{oy - v[1]:.2f}" for v in verts)
        return (f'<polygon points="{pts}" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')

    # RF exclusion zone
    rf_keepout = rf_union.buffer(GAP_UM, join_style="mitre", mitre_limit=10.0)
    if hasattr(rf_keepout, 'exterior'):
        lines.append(_svg_poly(list(rf_keepout.exterior.coords),
                               "none", stroke="#ff0000", sw=0.3, opacity=0.3))

    # RF polygons
    for verts in rf_polys_2d:
        lines.append(_svg_poly(verts, "#ff6666", sw=0.3))

    # DC polygons
    for verts in dc_polys:
        lines.append(_svg_poly(verts, "#6699ff", sw=0.3))

    # Cell boundary
    lines.append(f'<rect x="{ox - C}" y="{oy - C}" width="{2*C}" height="{2*C}" '
                 f'fill="none" stroke="#999" stroke-width="0.5" stroke-dasharray="4,2"/>')

    # Scale bar
    lines.append(f'<line x1="{ox+10}" y1="{oy+C+20}" x2="{ox+110}" y2="{oy+C+20}" '
                 f'stroke="black" stroke-width="1"/>')
    lines.append(f'<text x="{ox+60}" y="{oy+C+35}" text-anchor="middle" '
                 f'font-size="10" font-family="monospace">100 µm</text>')

    # Legend
    ly = 15
    for label, color in [("RF", "#ff6666"), ("DC", "#6699ff")]:
        lines.append(f'<rect x="5" y="{ly}" width="12" height="8" fill="{color}" '
                     f'stroke="#333" stroke-width="0.3"/>')
        lines.append(f'<text x="22" y="{ly+8}" font-size="9" '
                     f'font-family="monospace">{label}</text>')
        ly += 14

    lines.append('</svg>')
    Path(path).write_text('\n'.join(lines))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _min_rf_dc_distance(rf_union: ShapelyPolygon,
                        dc_polys: list[list[tuple[float, float]]]) -> float:
    if not dc_polys:
        return float('inf')
    dists = []
    for verts in dc_polys:
        dc = ShapelyPolygon(verts)
        dists.append(rf_union.distance(dc))
    return min(dists)


def _print_diagnostics(p: dict,
                       rf_polys_2d: list[list[tuple[float, float]]],
                       dc_polys: list[list[tuple[float, float]]],
                       rf_union: ShapelyPolygon,
                       rf_compound: TopoDS_Shape,
                       dc_compound: TopoDS_Shape | None) -> None:
    gi = p["center_gap_um"] / 2.0
    go = gi + p["rf_rail_width_um"]
    ji = p["junction_inner_um"]
    jo = p["junction_outer_um"]

    print(f"\n{'='*50}")
    print(f"RF-first surface junction generator")
    print(f"{'='*50}")
    print(f"  Derived: gi={gi:.1f}  go={go:.1f}  ji={ji:.1f}  jo={jo:.1f}")
    print(f"  RF top polygon count: {len(rf_polys_2d)}")
    print(f"  DC polygon count:     {len(dc_polys)}")

    rb = _shape_bbox(rf_compound)
    print(f"  RF bbox: x=[{rb[0]:.1f}, {rb[3]:.1f}]  y=[{rb[1]:.1f}, {rb[4]:.1f}]  "
          f"z=[{rb[2]:.1f}, {rb[5]:.1f}] µm")

    if dc_compound is not None:
        db = _shape_bbox(dc_compound)
        print(f"  DC bbox: x=[{db[0]:.1f}, {db[3]:.1f}]  y=[{db[1]:.1f}, {db[4]:.1f}]  "
              f"z=[{db[2]:.1f}, {db[5]:.1f}] µm")

    min_dist = _min_rf_dc_distance(rf_union, dc_polys)
    print(f"  Min RF–DC distance: {min_dist:.2f} µm")

    if min_dist < GAP_UM - 0.01:
        print(f"  WARNING: min RF–DC distance {min_dist:.2f} < {GAP_UM} µm!")

    # Polygon validity checks
    for i, verts in enumerate(rf_polys_2d):
        sp = ShapelyPolygon(verts)
        if not sp.is_valid:
            print(f"  WARNING: RF polygon {i} is invalid: {sp.is_valid}")
    for i, verts in enumerate(dc_polys):
        sp = ShapelyPolygon(verts)
        if not sp.is_valid:
            print(f"  WARNING: DC polygon {i} is invalid")

    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _extract_rf_poly_coords(rf_union) -> list[list[tuple[float, float]]]:
    """Extract polygon vertex lists from a Shapely geometry (Polygon or MultiPolygon)."""
    result: list[list[tuple[float, float]]] = []
    if isinstance(rf_union, ShapelyPolygon):
        coords = list(rf_union.exterior.coords)
        if coords[-1] == coords[0]:
            coords = coords[:-1]
        result.append(coords)
    elif isinstance(rf_union, MultiPolygon):
        for poly in rf_union.geoms:
            coords = list(poly.exterior.coords)
            if coords[-1] == coords[0]:
                coords = coords[:-1]
            result.append(coords)
    return result


def generate(p: dict) -> None:
    _validate(p)

    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build RF
    rf_union = _build_rf_shapely(p)
    rf_polys_2d = _rf_polygons_2d(p)

    # For STEP export, use the union polygons (handles curve rounding)
    if p["junction_curve_radius_um"] > 0:
        rf_export_polys = _extract_rf_poly_coords(rf_union)
    else:
        rf_export_polys = rf_polys_2d

    # Build DC
    dc_polys = _clip_dc(rf_union, p)

    # Extrude RF
    rf_shapes = [_extrude_polygon(v) for v in rf_export_polys]
    rf_compound = _make_compound(rf_shapes)

    # Extrude DC
    dc_shapes = [_extrude_polygon(v) for v in dc_polys]
    dc_compound = _make_compound(dc_shapes) if dc_shapes else None

    # Diagnostics
    _print_diagnostics(p, rf_export_polys, dc_polys, rf_union, rf_compound, dc_compound)

    # Write STEP files
    rf_path = str(out_dir / "rf.step")
    dc_path = str(out_dir / "dc.step")
    combined_path = str(out_dir / "combined.step")
    svg_path = str(out_dir / "debug_layout.svg")

    _write_step(rf_compound, rf_path)
    print(f"  Wrote {rf_path}")

    if dc_compound is not None:
        _write_step(dc_compound, dc_path)
        print(f"  Wrote {dc_path}")

        combined = _make_compound(rf_shapes + dc_shapes)
        _write_step(combined, combined_path)
        print(f"  Wrote {combined_path}")
    else:
        print("  WARNING: no DC polygons survived clipping — skipping dc.step / combined.step")

    # SVG
    _write_svg(rf_export_polys, dc_polys, rf_union, svg_path)
    print(f"  Wrote {svg_path}")

    # params.json
    params_out = {k: v for k, v in p.items() if k != "out_dir"}
    params_out["cell_half_um"] = CELL_HALF_UM
    params_out["cell_size_um"] = CELL_SIZE_UM
    params_out["gap_um"] = GAP_UM
    params_out["thickness_um"] = THICKNESS_UM
    params_path = str(out_dir / "params.json")
    Path(params_path).write_text(json.dumps(params_out, indent=2) + "\n")
    print(f"  Wrote {params_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="RF-first surface junction electrode generator")
    ap.add_argument("--rf-rail-width-um", type=float, default=60.0)
    ap.add_argument("--center-gap-um", type=float, default=100.0)
    ap.add_argument("--junction-inner-um", type=float, default=30.0)
    ap.add_argument("--junction-outer-um", type=float, default=50.0)
    ap.add_argument("--junction-curve-radius-um", type=float, default=0.0)
    ap.add_argument("--out-dir", type=str, default="cad/generated/rf_first_baseline")
    args = ap.parse_args()

    p = {
        "rf_rail_width_um": args.rf_rail_width_um,
        "center_gap_um": args.center_gap_um,
        "junction_inner_um": args.junction_inner_um,
        "junction_outer_um": args.junction_outer_um,
        "junction_curve_radius_um": args.junction_curve_radius_um,
        "out_dir": args.out_dir,
    }
    generate(p)


if __name__ == "__main__":
    main()
