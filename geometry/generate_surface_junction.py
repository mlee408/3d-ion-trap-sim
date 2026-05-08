#!/usr/bin/env python3
"""Parameterized surface-electrode X-junction generator (upper layer only).

Reproduces the upper-layer layout from the surface electrode structure.
RF is built as ONE unified connected polygon — no separate rail rectangles
or triangle caps.  The outer RF ring belongs to the lower layer and is NOT
generated here.

Corner L-shaped ground pads are fixed/sacred geometry — they are 3D RF beam
mounting points and must not be modified.

Electrode layers:  RF (red), DC (blue), Ground (green).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon as SPoly, MultiPolygon, box as sbox
from shapely.ops import unary_union
from shapely.validation import explain_validity

# ---------------------------------------------------------------------------
# Fixed constants
# ---------------------------------------------------------------------------
CELL_HALF = 300.0          # half of 600 µm unit cell
THICKNESS_UM = 10.0

# ---------------------------------------------------------------------------
# Default baseline values (extracted from SVG)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    cell_size_um=600.0,
    electrode_gap_um=10.0,
    o1_x_um=5.0,
    o1_y_um=23.07,
    o2_x_um=23.07,
    o2_y_um=5.0,
    o3_x_um=95.27,                # taper start along branch axis
    o3_y_um=60.0,                 # taper start perpendicular (= rf half-width)
    taper_rounding_um=0.0,
    dc_min_width_um=20.0,
    dc_segment_length_um=40.0,
    dc_segment_gap_um=10.0,
    dc_junction_keepout_um=0.0,
    ground_min_width_um=10.0,
)


# ---------------------------------------------------------------------------
# RF polygon builder — single connected body with C4 symmetry
# ---------------------------------------------------------------------------

def _build_rf_junction_quadrant(p: dict) -> list[tuple[float, float]]:
    """Return taper vertices for Q1 (+x, +y quadrant).

    Goes from the +x branch inner edge, through the taper, to the +y branch
    inner edge.  The full RF polygon is assembled by reflecting this across
    all four quadrants.

    Coordinate convention: center = (0,0), +x right, +y up.
    """
    o1x, o1y = p["o1_x_um"], p["o1_y_um"]
    o2x, o2y = p["o2_x_um"], p["o2_y_um"]
    o3x, o3y = p["o3_x_um"], p["o3_y_um"]

    # Inner-edge taper path from +x branch to +y branch (Q1):
    #   o3 → o2 → o1 → o3_mirror
    return [
        (o3x, o3y),    # taper start on +x branch inner edge
        (o2x, o2y),
        (o1x, o1y),
        (o3y, o3x),    # taper end on +y branch inner edge (swapped)
    ]


def build_rf_polygon(p: dict) -> SPoly:
    """Build the unified RF cross polygon as a single connected body.

    The RF cross has 120 µm wide branches extending ±300 µm from center,
    connected by an intricate taper at the junction.  The outer RF ring
    (unit cell boundary) is returned separately by build_rf_ring().
    """
    hw = p["o3_y_um"]  # RF half-width = o3's perpendicular component
    C = CELL_HALF

    q1 = _build_rf_junction_quadrant(p)

    def rot90(pts):
        return [(-y, x) for x, y in pts]
    def rot180(pts):
        return [(-x, -y) for x, y in pts]
    def rot270(pts):
        return [(y, -x) for x, y in pts]

    q2 = rot90(q1)
    q3 = rot180(q1)
    q4 = rot270(q1)

    cross_pts = []
    cross_pts.append((-C, -hw))
    cross_pts.extend(q3)
    cross_pts.append((-hw, -C))
    cross_pts.append((hw, -C))
    cross_pts.extend(q4)
    cross_pts.append((C, -hw))
    cross_pts.append((C, hw))
    cross_pts.extend(q1)
    cross_pts.append((hw, C))
    cross_pts.append((-hw, C))
    cross_pts.extend(q2)
    cross_pts.append((-C, hw))

    cross = SPoly(cross_pts)
    if not cross.is_valid:
        cross = cross.buffer(0)

    r = p["taper_rounding_um"]
    if r > 0:
        cross = cross.buffer(r, resolution=64).buffer(-r, resolution=64)

    return cross


# ---------------------------------------------------------------------------
# DC electrode builder — segments from baseline SVG topology
# ---------------------------------------------------------------------------

def _baseline_dc_candidates(p: dict) -> list[list[tuple[float, float]]]:
    """Return DC candidate rectangles matching the baseline SVG topology.

    Defines segments for the +y branch, +x side, then applies C4 rotation
    and mirror symmetry to generate all 36 DC segments.  Candidate rectangles
    are later clipped against the RF keepout to produce final DC shapes
    (including chamfered corners near the junction).
    """
    hw = p["o3_y_um"]
    gap = p["electrode_gap_um"]
    C = CELL_HALF
    seg_gap = p["dc_segment_gap_um"]

    dc_inner = hw + gap        # 70 µm from center axis
    dc_outer = dc_inner + 60   # 130 µm — DC electrode depth perpendicular to branch
    jpc_extent = 100            # junction corner pad extent

    # --- Template: +y branch, +x side (4 segments) ---
    template_segs = [
        (dc_inner, dc_outer, 260, 295),
        (dc_inner, dc_outer, 210, 250),
        (dc_inner, dc_outer, 160, 200),
    ]
    # Innermost segment: start as full rectangle, then cut a 45° chamfer only
    # at the corner closest to the adjacent branch's mirrored segment.
    # The two adjacent chamfers lie on lines y = x + k; gap = sqrt(2)*|k|.
    seg_y_start = jpc_extent + seg_gap   # 110 µm
    seg_y_end = 150
    # The 45° cut line passes through (chamfer_x, seg_y_start) going to
    # (dc_outer, chamfer_y). We only need to cut the triangle that violates
    # the gap constraint with the rotated neighbor.
    min_offset = math.ceil(gap / math.sqrt(2))
    chamfer_x = seg_y_start - min_offset
    chamfer_x = max(chamfer_x, dc_inner)
    chamfer_y = seg_y_start + dc_outer - chamfer_x
    if chamfer_x >= dc_outer:
        chamfer_seg = rect(dc_inner, dc_outer, seg_y_start, seg_y_end)
    elif chamfer_y <= seg_y_end:
        chamfer_seg = [
            (dc_inner, seg_y_start), (chamfer_x, seg_y_start),
            (dc_outer, chamfer_y), (dc_outer, seg_y_end), (dc_inner, seg_y_end),
        ]
    else:
        exit_x = chamfer_x + (seg_y_end - seg_y_start)
        exit_x = min(exit_x, dc_outer)
        chamfer_seg = [
            (dc_inner, seg_y_start), (chamfer_x, seg_y_start),
            (exit_x, seg_y_end), (dc_inner, seg_y_end),
        ]

    # --- Junction corner pad template (Q1: +x, +y quadrant) ---
    junction_pad = (gap, jpc_extent, gap, jpc_extent)

    def rect(x0, x1, y0, y1):
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def rot90(pts):
        return [(-y, x) for x, y in pts]

    def mirror_x(pts):
        return [(-x, y) for x, y in pts]

    candidates = []

    # Generate for all 8 branch-sides using C4 rotation
    base_rects = [rect(*s) for s in template_segs]
    base_rects.append(chamfer_seg)

    for rotation in range(4):
        pos_side = base_rects
        neg_side = [mirror_x(r) for r in base_rects]

        for r in pos_side + neg_side:
            rotated = r
            for _ in range(rotation):
                rotated = rot90(rotated)
            candidates.append(rotated)

    # Junction corner pads (4 quadrants)
    jp = rect(*junction_pad)
    for rotation in range(4):
        rotated = jp
        for _ in range(rotation):
            rotated = rot90(rotated)
        candidates.append(rotated)

    return candidates


def build_dc_polygons(p: dict, rf_poly: SPoly) -> list[list[tuple[float, float]]]:
    """Build DC electrode polygons by clipping candidate regions against RF keepout."""
    gap = p["electrode_gap_um"]
    min_w = p["dc_min_width_um"]

    rf_keepout = rf_poly.buffer(gap, join_style="mitre", mitre_limit=10.0)

    candidates = _baseline_dc_candidates(p)
    result = []

    for cand_verts in candidates:
        cand = SPoly(cand_verts)
        if not cand.is_valid:
            cand = cand.buffer(0)
        if cand.area < 1.0:
            continue

        clipped = cand.difference(rf_keepout)
        if clipped.is_empty:
            continue

        polys = [clipped] if isinstance(clipped, SPoly) else list(clipped.geoms)
        for poly in polys:
            if poly.is_empty or poly.area < min_w * min_w:
                continue
            # Check minimum width using negative buffer
            shrunk = poly.buffer(-min_w / 2.0)
            if shrunk.is_empty:
                continue
            coords = list(poly.exterior.coords)
            if coords[-1] == coords[0]:
                coords = coords[:-1]
            if len(coords) >= 3:
                result.append(coords)

    return result


# ---------------------------------------------------------------------------
# Ground fill
# ---------------------------------------------------------------------------

def build_ground_polygons(p: dict, rf_poly: SPoly,
                          dc_polys: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Build ground corner pads for the upper layer.

    Each pad is an L-shaped region in one quadrant with a square cutout at
    the corner — the cutout is where 3D RF beams are placed.  The outer
    boundary fills the inter-branch area; inner edges are clipped against
    RF and DC keepout zones so they shrink automatically when parameters
    change.
    """
    gap = p["electrode_gap_um"]
    min_w = p["ground_min_width_um"]
    C = CELL_HALF
    hw = p["o3_y_um"]
    dc_inner = hw + gap
    dc_outer = dc_inner + 60
    gi = dc_outer + gap

    cutout = 40.0
    corner_gnd = [
        # Q1: +x, +y
        [(C, C - cutout), (C, gi), (gi, gi), (gi, C), (C - cutout, C), (C - cutout, C - cutout)],
        # Q2: -x, +y
        [(-C, C - cutout), (-C + cutout, C - cutout), (-C + cutout, C), (-gi, C), (-gi, gi), (-C, gi)],
        # Q3: -x, -y
        [(-C + cutout, -C), (-C + cutout, -C + cutout), (-C, -C + cutout), (-C, -gi), (-gi, -gi), (-gi, -C)],
        # Q4: +x, -y
        [(C, -C + cutout), (C - cutout, -C + cutout), (C - cutout, -C), (gi, -C), (gi, -gi), (C, -gi)],
    ]

    rf_keepout = rf_poly.buffer(gap, join_style="mitre", mitre_limit=10.0)
    dc_shapes = [SPoly(v) for v in dc_polys if len(v) >= 3]
    dc_keepout = unary_union([s.buffer(gap, join_style="mitre", mitre_limit=10.0)
                              for s in dc_shapes]) if dc_shapes else SPoly()
    total_keepout = unary_union([rf_keepout, dc_keepout])

    result = []
    for verts in corner_gnd:
        gnd = SPoly(verts)
        if not gnd.is_valid:
            gnd = gnd.buffer(0)
        clipped = gnd.difference(total_keepout)
        _collect_polys(clipped, result, min_w)

    return result


def _collect_polys(geom, result: list, min_w: float):
    if geom.is_empty:
        return
    polys = [geom] if isinstance(geom, SPoly) else list(geom.geoms)
    for poly in polys:
        if poly.is_empty or poly.area < min_w * min_w:
            continue
        coords = list(poly.exterior.coords)
        if coords[-1] == coords[0]:
            coords = coords[:-1]
        if len(coords) >= 3:
            result.append(coords)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(p: dict, rf_poly: SPoly,
             dc_polys: list[list[tuple[float, float]]],
             gnd_polys: list[list[tuple[float, float]]]) -> list[str]:
    """Run all validation checks. Returns list of errors (empty = pass)."""
    errors = []
    gap = p["electrode_gap_um"]
    dc_min_w = p["dc_min_width_um"]
    gnd_min_w = p["ground_min_width_um"]

    # RF: single polygon, valid
    if isinstance(rf_poly, MultiPolygon):
        errors.append(f"RF is MultiPolygon with {len(rf_poly.geoms)} parts — must be 1")
    if not rf_poly.is_valid:
        errors.append(f"RF polygon invalid: {explain_validity(rf_poly)}")

    # RF-DC gap
    for i, verts in enumerate(dc_polys):
        dc = SPoly(verts)
        d = rf_poly.distance(dc)
        if d < gap - 0.5:
            errors.append(f"RF-DC[{i}] gap = {d:.2f} < {gap} µm")

    # DC-DC gap
    dc_shapes = [SPoly(v) for v in dc_polys]
    for i in range(len(dc_shapes)):
        for j in range(i + 1, len(dc_shapes)):
            d = dc_shapes[i].distance(dc_shapes[j])
            if d < gap - 0.5 and d > 0.01:
                errors.append(f"DC[{i}]-DC[{j}] gap = {d:.2f} < {gap} µm")

    # RF-GND gap
    for i, verts in enumerate(gnd_polys):
        gnd = SPoly(verts)
        d = rf_poly.distance(gnd)
        if d < gap - 0.5:
            errors.append(f"RF-GND[{i}] gap = {d:.2f} < {gap} µm")

    # DC-GND gap
    gnd_shapes = [SPoly(v) for v in gnd_polys]
    for i, dc in enumerate(dc_shapes):
        for j, gnd in enumerate(gnd_shapes):
            d = dc.distance(gnd)
            if d < gap - 0.5 and d > 0.01:
                errors.append(f"DC[{i}]-GND[{j}] gap = {d:.2f} < {gap} µm")

    # DC min width
    for i, verts in enumerate(dc_polys):
        dc = SPoly(verts)
        shrunk = dc.buffer(-dc_min_w / 2.0)
        if shrunk.is_empty:
            errors.append(f"DC[{i}] narrower than {dc_min_w} µm")

    # C4 symmetry check (approximate)
    rf_area = rf_poly.area
    for angle in [90, 180, 270]:
        from shapely.affinity import rotate
        rotated = rotate(rf_poly, angle, origin=(0, 0))
        sym_diff = rf_poly.symmetric_difference(rotated).area
        if sym_diff / rf_area > 0.01:
            errors.append(f"RF C4 symmetry broken at {angle}°: diff={sym_diff:.1f} / {rf_area:.1f}")

    return errors


# ---------------------------------------------------------------------------
# SVG export
# ---------------------------------------------------------------------------

def write_svg(rf_poly: SPoly,
              dc_polys: list[list[tuple[float, float]]],
              gnd_polys: list[list[tuple[float, float]]],
              path: str) -> None:
    C = CELL_HALF
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

    def svg_poly(verts, fill, stroke="#333", sw=0.3, opacity=0.8):
        pts = " ".join(f"{ox + v[0]:.2f},{oy - v[1]:.2f}" for v in verts)
        return (f'<polygon points="{pts}" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')

    # Ground (draw first, behind)
    for verts in gnd_polys:
        lines.append(svg_poly(verts, "#88cc88"))

    # RF — use SVG path for polygon with holes
    def svg_path_ring(exterior_coords, interior_rings, fill):
        def ring_to_d(coords, close=True):
            parts = []
            for i, (x, y) in enumerate(coords):
                sx, sy = ox + x, oy - y
                cmd = "M" if i == 0 else "L"
                parts.append(f"{cmd}{sx:.2f},{sy:.2f}")
            if close:
                parts.append("Z")
            return " ".join(parts)
        d = ring_to_d(exterior_coords)
        for ring in interior_rings:
            d += " " + ring_to_d(ring)
        return (f'<path d="{d}" fill="{fill}" fill-rule="evenodd" '
                f'stroke="#333" stroke-width="0.3" opacity="0.8"/>')

    ext = list(rf_poly.exterior.coords)
    if ext[-1] == ext[0]:
        ext = ext[:-1]
    holes = []
    for interior in rf_poly.interiors:
        h = list(interior.coords)
        if h[-1] == h[0]:
            h = h[:-1]
        holes.append(h)
    lines.append(svg_path_ring(ext, holes, "#ff6666"))

    # DC
    for verts in dc_polys:
        lines.append(svg_poly(verts, "#6699ff"))

    # Scale bar
    lines.append(f'<line x1="{ox}" y1="{oy + C + 20}" '
                 f'x2="{ox + 100}" y2="{oy + C + 20}" '
                 f'stroke="black" stroke-width="1"/>')
    lines.append(f'<text x="{ox + 50}" y="{oy + C + 35}" text-anchor="middle" '
                 f'font-size="10" font-family="monospace">100 µm</text>')

    # Legend
    ly = 15
    for label, color in [("RF", "#ff6666"), ("DC", "#6699ff"), ("Ground", "#88cc88")]:
        lines.append(f'<rect x="5" y="{ly}" width="12" height="8" fill="{color}" '
                     f'stroke="#333" stroke-width="0.3"/>')
        lines.append(f'<text x="22" y="{ly + 8}" font-size="9" '
                     f'font-family="monospace">{label}</text>')
        ly += 14

    lines.append('</svg>')
    Path(path).write_text('\n'.join(lines))


# ---------------------------------------------------------------------------
# PNG export
# ---------------------------------------------------------------------------

def write_png(rf_poly: SPoly,
              dc_polys: list[list[tuple[float, float]]],
              gnd_polys: list[list[tuple[float, float]]],
              path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MPoly
        from matplotlib.collections import PatchCollection
    except ImportError:
        print("  matplotlib not available — skipping PNG export")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    C = CELL_HALF

    def add_polys(verts_list, color, label=None):
        patches = []
        for verts in verts_list:
            patches.append(MPoly(verts, closed=True))
        if patches:
            pc = PatchCollection(patches, facecolor=color, edgecolor="#333",
                                 linewidth=0.3, alpha=0.8, label=label)
            ax.add_collection(pc)

    # Ground
    add_polys(gnd_polys, "#88cc88", "Ground")

    # RF cross
    rf_coords = list(rf_poly.exterior.coords)
    add_polys([rf_coords], "#ff6666", "RF")

    # DC
    add_polys(dc_polys, "#6699ff", "DC")

    ax.set_xlim(-C - 40, C + 40)
    ax.set_ylim(-C - 40, C + 40)
    ax.set_aspect("equal")
    ax.set_title("Surface junction layout")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.grid(True, alpha=0.3)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# STEP export (optional, requires OCC)
# ---------------------------------------------------------------------------

def write_step(rf_poly: SPoly,
               dc_polys: list[list[tuple[float, float]]],
               gnd_polys: list[list[tuple[float, float]]],
               out_dir: Path,
               p: dict | None = None) -> None:
    try:
        from OCP.gp import gp_Pnt, gp_Vec
        from OCP.BRepBuilderAPI import (
            BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace)
        from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
        from OCP.TopoDS import TopoDS_Compound
        from OCP.BRep import BRep_Builder
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.Interface import Interface_Static
    except ImportError:
        print("  OCP (cadquery) not available — skipping STEP export")
        return

    def mm(x):
        return x * 1e-3

    def extrude(verts, thickness_um=THICKNESS_UM):
        n = len(verts)
        wb = BRepBuilderAPI_MakeWire()
        for i in range(n):
            x0, y0 = verts[i]
            x1, y1 = verts[(i + 1) % n]
            edge = BRepBuilderAPI_MakeEdge(
                gp_Pnt(mm(x0), mm(y0), mm(-thickness_um)),
                gp_Pnt(mm(x1), mm(y1), mm(-thickness_um))).Edge()
            wb.Add(edge)
        face = BRepBuilderAPI_MakeFace(wb.Wire(), True).Face()
        return BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, mm(thickness_um))).Shape()

    def make_compound(shapes):
        b = BRep_Builder()
        c = TopoDS_Compound()
        b.MakeCompound(c)
        for s in shapes:
            b.Add(c, s)
        return c

    def write(shape, path):
        w = STEPControl_Writer()
        Interface_Static.SetCVal_s("write.step.schema", "AP203")
        w.Transfer(shape, STEPControl_AsIs)
        if w.Write(str(path)) != 1:
            raise RuntimeError(f"STEP write failed: {path}")

    # RF — uniform extrusion
    rf_coords = list(rf_poly.exterior.coords)
    if rf_coords[-1] == rf_coords[0]:
        rf_coords = rf_coords[:-1]
    rf_shape = extrude(rf_coords)
    write(rf_shape, out_dir / "rf.step")
    print(f"  Wrote {out_dir / 'rf.step'}")

    # DC
    if dc_polys:
        dc_shapes = [extrude(v) for v in dc_polys]
        write(make_compound(dc_shapes), out_dir / "dc.step")
        print(f"  Wrote {out_dir / 'dc.step'}")

    # Ground
    if gnd_polys:
        gnd_shapes = [extrude(v) for v in gnd_polys]
        write(make_compound(gnd_shapes), out_dir / "ground.step")
        print(f"  Wrote {out_dir / 'ground.step'}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate(p: dict) -> dict:
    """Generate all electrodes. Returns dict with polygons and validation."""
    print(f"\n{'='*60}")
    print(f"Surface junction generator")
    print(f"{'='*60}")
    print(f"  Electrode gap: {p['electrode_gap_um']} µm")
    print(f"  O1: ({p['o1_x_um']}, {p['o1_y_um']})")
    print(f"  O2: ({p['o2_x_um']}, {p['o2_y_um']})")
    print(f"  O3: ({p['o3_x_um']}, {p['o3_y_um']})  [RF half-width={p['o3_y_um']}, taper start={p['o3_x_um']}]")
    print(f"  Taper rounding: {p['taper_rounding_um']} µm")

    # Build RF cross (upper layer only — no ring)
    rf_poly = build_rf_polygon(p)
    print(f"\n  RF cross: area={rf_poly.area:.1f} µm², "
          f"valid={rf_poly.is_valid}, type={rf_poly.geom_type}")

    # Build DC (clipped against RF cross only; ring is boundary)
    dc_polys = build_dc_polygons(p, rf_poly)
    print(f"  DC segments: {len(dc_polys)}")

    # Build ground (corner L-shapes, clipped against RF+DC keepout)
    gnd_polys = build_ground_polygons(p, rf_poly, dc_polys)
    print(f"  Ground regions: {len(gnd_polys)} (corner pads)")

    # Validate
    do_validate = p.get("validate", False)
    errors = validate(p, rf_poly, dc_polys, gnd_polys)
    if errors:
        print(f"\n  VALIDATION ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
    else:
        print(f"\n  All validation checks passed.")

    # Export
    out_dir = Path(p.get("out_dir", "cad/generated/surface_junction"))
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = p.get("out_svg", str(out_dir / "layout.svg"))
    write_svg(rf_poly, dc_polys, gnd_polys, svg_path)
    print(f"  Wrote {svg_path}")

    png_path = p.get("out_png", str(out_dir / "layout.png"))
    write_png(rf_poly, dc_polys, gnd_polys, png_path)

    write_step(rf_poly, dc_polys, gnd_polys, out_dir, p=p)

    # Params
    params_out = {k: v for k, v in p.items()
                  if k not in ("out_dir", "out_svg", "out_png", "validate", "export_step")}
    params_path = str(out_dir / "params.json")
    Path(params_path).write_text(json.dumps(params_out, indent=2) + "\n")
    print(f"  Wrote {params_path}")

    if errors and do_validate:
        sys.exit(1)

    return {"rf_poly": rf_poly, "dc_polys": dc_polys,
            "gnd_polys": gnd_polys, "errors": errors}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parameterized surface-electrode X-junction generator")

    ap.add_argument("--input-svg", type=str, default=None,
                    help="Baseline SVG for reference (not parsed, used for docs)")
    ap.add_argument("--electrode-gap-um", type=float, default=DEFAULTS["electrode_gap_um"])
    ap.add_argument("--o1-x-um", type=float, default=DEFAULTS["o1_x_um"])
    ap.add_argument("--o1-y-um", type=float, default=DEFAULTS["o1_y_um"])
    ap.add_argument("--o2-x-um", type=float, default=DEFAULTS["o2_x_um"])
    ap.add_argument("--o2-y-um", type=float, default=DEFAULTS["o2_y_um"])
    ap.add_argument("--o3-x-um", type=float, default=DEFAULTS["o3_x_um"],
                    help="Taper start along branch axis (was junction-extent)")
    ap.add_argument("--o3-y-um", type=float, default=DEFAULTS["o3_y_um"],
                    help="RF half-width (was rf-width/2)")
    ap.add_argument("--taper-rounding-um", type=float, default=DEFAULTS["taper_rounding_um"])
    ap.add_argument("--dc-min-width-um", type=float, default=DEFAULTS["dc_min_width_um"])
    ap.add_argument("--dc-segment-length-um", type=float, default=DEFAULTS["dc_segment_length_um"])
    ap.add_argument("--dc-segment-gap-um", type=float, default=DEFAULTS["dc_segment_gap_um"])
    ap.add_argument("--dc-junction-keepout-um", type=float,
                    default=DEFAULTS["dc_junction_keepout_um"])
    ap.add_argument("--ground-min-width-um", type=float, default=DEFAULTS["ground_min_width_um"])
    ap.add_argument("--out-dir", type=str, default="cad/generated/surface_junction")
    ap.add_argument("--out-svg", type=str, default=None)
    ap.add_argument("--out-png", type=str, default=None)
    ap.add_argument("--validate", action="store_true")

    args = ap.parse_args()

    p = {
        "cell_size_um": 600.0,
        "electrode_gap_um": args.electrode_gap_um,
        "o1_x_um": args.o1_x_um,
        "o1_y_um": args.o1_y_um,
        "o2_x_um": args.o2_x_um,
        "o2_y_um": args.o2_y_um,
        "o3_x_um": args.o3_x_um,
        "o3_y_um": args.o3_y_um,
        "taper_rounding_um": args.taper_rounding_um,
        "dc_min_width_um": args.dc_min_width_um,
        "dc_segment_length_um": args.dc_segment_length_um,
        "dc_segment_gap_um": args.dc_segment_gap_um,
        "dc_junction_keepout_um": args.dc_junction_keepout_um,
        "ground_min_width_um": args.ground_min_width_um,
        "out_dir": args.out_dir,
        "validate": args.validate,
    }

    if args.out_svg:
        p["out_svg"] = args.out_svg
    if args.out_png:
        p["out_png"] = args.out_png

    generate(p)


if __name__ == "__main__":
    main()
