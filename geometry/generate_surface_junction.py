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

# Lower layer (constant geometry — structural base plane)
LOWER_THICKNESS_UM = 10.0          # z = [-20, -10] µm
RF_RING_OUTER_UM   = 328.0         # 3D lattice beam half-extent (656/2)
GND_PLANE_HALF_UM  = 272.0         # support_half − beam_half (300 − 28)

# ---------------------------------------------------------------------------
# Default baseline values (extracted from SVG)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    cell_size_um=600.0,
    electrode_gap_um=10.0,
    o1_x_um=5.0,                  # neck start (near center)
    o1_y_um=5.0,                  # neck half-width
    o1_r_um=0.0,
    o2_x_um=23.07,               # neck end (taper begins)
    o2_y_um=5.0,                  # same y as o1
    o2_r_um=0.0,
    o3_x_um=95.27,                # taper start along branch axis
    o3_y_um=60.0,                 # taper start perpendicular (= rf half-width)
    o3_r_um=0.0,                  # curvature radius at o3
    dc_min_width_um=20.0,
    dc_segment_length_um=40.0,
    dc_segment_gap_um=10.0,
    dc_junction_keepout_um=0.0,
    ground_min_width_um=10.0,
)


# ---------------------------------------------------------------------------
# RF polygon builder — single connected body with C4 symmetry
# ---------------------------------------------------------------------------

def _arc_round(prev, corner, nxt, radius, n_pts=16):
    """Replace a sharp corner with a circular arc of given radius.

    Returns a list of points along the arc from the tangent point on
    prev→corner to the tangent point on corner→nxt.  If radius is 0
    or the geometry is degenerate, returns [corner].
    """
    if radius <= 0:
        return [corner]
    px, py = prev
    cx, cy = corner
    nx, ny = nxt
    d1x, d1y = px - cx, py - cy
    d2x, d2y = nx - cx, ny - cy
    len1 = math.hypot(d1x, d1y)
    len2 = math.hypot(d2x, d2y)
    if len1 < 1e-6 or len2 < 1e-6:
        return [corner]
    u1x, u1y = d1x / len1, d1y / len1
    u2x, u2y = d2x / len2, d2y / len2
    dot = u1x * u2x + u1y * u2y
    dot = max(-1.0, min(1.0, dot))
    half_angle = math.acos(dot) / 2.0
    if half_angle < 1e-6:
        return [corner]
    tan_len = min(radius / math.tan(half_angle), len1 * 0.45, len2 * 0.45)
    t1 = (cx + u1x * tan_len, cy + u1y * tan_len)
    t2 = (cx + u2x * tan_len, cy + u2y * tan_len)
    actual_r = tan_len * math.tan(half_angle)
    bis_x, bis_y = u1x + u2x, u1y + u2y
    bis_len = math.hypot(bis_x, bis_y)
    if bis_len < 1e-6:
        return [corner]
    bis_x, bis_y = bis_x / bis_len, bis_y / bis_len
    center_dist = actual_r / math.sin(half_angle)
    arc_cx = cx + bis_x * center_dist
    arc_cy = cy + bis_y * center_dist
    a_start = math.atan2(t1[1] - arc_cy, t1[0] - arc_cx)
    a_end = math.atan2(t2[1] - arc_cy, t2[0] - arc_cx)
    cross = (t1[0] - arc_cx) * (t2[1] - arc_cy) - (t1[1] - arc_cy) * (t2[0] - arc_cx)
    if cross < 0:
        if a_end > a_start:
            a_end -= 2 * math.pi
    else:
        if a_end < a_start:
            a_end += 2 * math.pi
    pts = []
    for i in range(n_pts + 1):
        t = i / n_pts
        a = a_start + t * (a_end - a_start)
        pts.append((arc_cx + actual_r * math.cos(a),
                     arc_cy + actual_r * math.sin(a)))
    return pts


def _build_rf_junction_quadrant(p: dict) -> list[tuple[float, float]]:
    """Return taper vertices for Q1 (+x, +y quadrant).

    Path: o3 → o2 → o1 → o1_mirror → o2_mirror → o3_mirror
    where mirror swaps (x,y)→(y,x).

    o1: start of neck (near center), o2: end of neck (taper begins),
    o3: taper ends at straight branch.  Neck runs at constant y = o1_y = o2_y.
    """
    o1 = (p["o1_x_um"], p["o1_y_um"])
    o2 = (p["o2_x_um"], p["o2_y_um"])
    o3 = (p["o3_x_um"], p["o3_y_um"])
    o1m = (o1[1], o1[0])
    o2m = (o2[1], o2[0])
    o3m = (o3[1], o3[0])
    r1, r2, r3 = p["o1_r_um"], p["o2_r_um"], p["o3_r_um"]

    pts = []
    pts.append(o3)
    if abs(o1[0] - o1[1]) < 0.01:
        # o1 sits on the diagonal — o1 and o1_mirror collapse to one point
        pts.extend(_arc_round(o3, o2, o1, r2))
        pts.extend(_arc_round(o2, o1, o2m, r1))
        pts.extend(_arc_round(o1, o2m, o3m, r2))
    else:
        pts.extend(_arc_round(o3, o2, o1, r2))
        pts.extend(_arc_round(o2, o1, o1m, r1))
        pts.extend(_arc_round(o1, o1m, o2m, r1))
        pts.extend(_arc_round(o1m, o2m, o3m, r2))
    pts.append(o3m)
    return pts


def build_rf_polygon(p: dict) -> SPoly:
    """Build the unified RF cross polygon as a single connected body.

    The RF cross has branches extending ±300 µm from center, connected by
    a taper at the junction defined by control points o1, o2, o3.
    """
    hw = p["o3_y_um"]
    C = CELL_HALF
    r3 = p["o3_r_um"]
    o2 = (p["o2_x_um"], p["o2_y_um"])
    o1 = (p["o1_x_um"], p["o1_y_um"])
    o3 = (p["o3_x_um"], p["o3_y_um"])
    o3m = (o3[1], o3[0])

    q1_inner = _build_rf_junction_quadrant(p)

    def rot90(pts):
        return [(-y, x) for x, y in pts]
    def rot180(pts):
        return [(-x, -y) for x, y in pts]
    def rot270(pts):
        return [(y, -x) for x, y in pts]

    def _build_branch_with_o3_rounding(branch_start, q_pts, branch_end):
        """Insert o3 arc rounding at the two branch-taper transitions."""
        pts = []
        pts.append(branch_start)
        first_taper = q_pts[0]
        second_after = q_pts[1] if len(q_pts) > 1 else branch_end
        pts.extend(_arc_round(branch_start, first_taper, second_after, r3))
        pts.extend(q_pts[1:-1])
        last_taper = q_pts[-1]
        second_before = q_pts[-2] if len(q_pts) > 1 else branch_start
        pts.extend(_arc_round(second_before, last_taper, branch_end, r3))
        pts.append(branch_end)
        return pts

    q1 = _build_branch_with_o3_rounding((C, hw), q1_inner, (hw, C))
    q2 = _build_branch_with_o3_rounding((-hw, C), rot90(q1_inner), (-C, hw))
    q3 = _build_branch_with_o3_rounding((-C, -hw), rot180(q1_inner), (-hw, -C))
    q4 = _build_branch_with_o3_rounding((hw, -C), rot270(q1_inner), (C, -hw))

    cross_pts = []
    cross_pts.extend(q3)
    cross_pts.extend(q4)
    cross_pts.extend(q1)
    cross_pts.extend(q2)

    cross = SPoly(cross_pts)
    if not cross.is_valid:
        cross = cross.buffer(0)

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
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
        from OCP.TopoDS import TopoDS_Compound
        from OCP.BRep import BRep_Builder
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.Interface import Interface_Static
    except ImportError:
        print("  OCP (cadquery) not available — skipping STEP export")
        return

    gap_um = p["electrode_gap_um"] if p else 10.0

    def mm(x):
        return x * 1e-3

    def make_box(x0, y0, z0, x1, y1, z1):
        """Create a solid box from two corner points (µm)."""
        verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        n = len(verts)
        wb = BRepBuilderAPI_MakeWire()
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % n]
            edge = BRepBuilderAPI_MakeEdge(
                gp_Pnt(mm(ax), mm(ay), mm(z0)),
                gp_Pnt(mm(bx), mm(by), mm(z0))).Edge()
            wb.Add(edge)
        face = BRepBuilderAPI_MakeFace(wb.Wire(), True).Face()
        return BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, mm(z1 - z0))).Shape()

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

    # --- Lower layer constants ---
    lower_z_bot = -(THICKNESS_UM + LOWER_THICKNESS_UM)   # -20 µm
    lower_z_top = -THICKNESS_UM                           # -10 µm
    ro = RF_RING_OUTER_UM                                 # 328 µm
    ri = GND_PLANE_HALF_UM + gap_um                       # 282 µm
    gh = GND_PLANE_HALF_UM                                # 272 µm

    # RF lower ring: outer box minus inner cutout
    rf_ring_outer = make_box(-ro, -ro, lower_z_bot, ro, ro, lower_z_top)
    rf_ring_cutout = make_box(-ri, -ri, lower_z_bot, ri, ri, lower_z_top)
    rf_lower_ring = BRepAlgoAPI_Cut(rf_ring_outer, rf_ring_cutout).Shape()

    # Ground lower plane
    gnd_lower_plane = make_box(-gh, -gh, lower_z_bot, gh, gh, lower_z_top)

    # RF — upper cross fused with lower ring
    rf_coords = list(rf_poly.exterior.coords)
    if rf_coords[-1] == rf_coords[0]:
        rf_coords = rf_coords[:-1]
    rf_upper = extrude(rf_coords)
    rf_shape = BRepAlgoAPI_Fuse(rf_upper, rf_lower_ring).Shape()
    write(rf_shape, out_dir / "rf.step")
    print(f"  Wrote {out_dir / 'rf.step'}")

    # DC — upper layer only
    dc_shapes = []
    if dc_polys:
        dc_shapes = [extrude(v) for v in dc_polys]
        write(make_compound(dc_shapes), out_dir / "dc.step")
        print(f"  Wrote {out_dir / 'dc.step'}")

    # Ground — upper pads + lower plane
    gnd_shapes = []
    if gnd_polys:
        gnd_shapes = [extrude(v) for v in gnd_polys]
    gnd_shapes.append(gnd_lower_plane)
    write(make_compound(gnd_shapes), out_dir / "ground.step")
    print(f"  Wrote {out_dir / 'ground.step'}")

    # Combined
    all_shapes = [rf_shape]
    all_shapes.extend(dc_shapes)
    all_shapes.extend(gnd_shapes)
    write(make_compound(all_shapes), out_dir / "combined.step")
    print(f"  Wrote {out_dir / 'combined.step'}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate(p: dict) -> dict:
    """Generate all electrodes. Returns dict with polygons and validation."""
    print(f"\n{'='*60}")
    print(f"Surface junction generator")
    print(f"{'='*60}")
    print(f"  Electrode gap: {p['electrode_gap_um']} µm")
    print(f"  O1: ({p['o1_x_um']}, {p['o1_y_um']})  r={p['o1_r_um']}")
    print(f"  O2: ({p['o2_x_um']}, {p['o2_y_um']})  r={p['o2_r_um']}")
    print(f"  O3: ({p['o3_x_um']}, {p['o3_y_um']})  r={p['o3_r_um']}  [RF half-width={p['o3_y_um']}, taper start={p['o3_x_um']}]")

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
    ap.add_argument("--o1-r-um", type=float, default=DEFAULTS["o1_r_um"])
    ap.add_argument("--o2-x-um", type=float, default=DEFAULTS["o2_x_um"])
    ap.add_argument("--o2-y-um", type=float, default=DEFAULTS["o2_y_um"])
    ap.add_argument("--o2-r-um", type=float, default=DEFAULTS["o2_r_um"])
    ap.add_argument("--o3-x-um", type=float, default=DEFAULTS["o3_x_um"],
                    help="Taper start along branch axis")
    ap.add_argument("--o3-y-um", type=float, default=DEFAULTS["o3_y_um"],
                    help="RF half-width")
    ap.add_argument("--o3-r-um", type=float, default=DEFAULTS["o3_r_um"])
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
        "o1_r_um": args.o1_r_um,
        "o2_x_um": args.o2_x_um,
        "o2_y_um": args.o2_y_um,
        "o2_r_um": args.o2_r_um,
        "o3_x_um": args.o3_x_um,
        "o3_y_um": args.o3_y_um,
        "o3_r_um": args.o3_r_um,
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
