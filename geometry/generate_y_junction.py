#!/usr/bin/env python3
"""Parameterized Y-junction surface electrode generator.

Loads baseline geometry from base STEP files (cad/base/Yjunction/).
Only the junction center region is parameterized; arm/extension
geometry comes from the loaded baseline.

Primary control variables (4):
  o1_y — neck half-width at junction center
  o1_r — curvature radius at neck
  o2_x — distance from center where RF reaches full arm width (taper length)
  o2_r — curvature radius where taper meets the straight arm

The modified RF surface is combined with rf_3d_y.step (structural
beams) for STEP output. DC electrodes are derived from baseline
distances and clipped against RF keepout. No ground electrode.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon as SPoly, MultiPolygon
from shapely.ops import unary_union

# ── Constants ────────────────────────────────────────────────────────────
THICKNESS_UM = 10.0
SURFACE_Y_BOT_UM = 30.0
BASE_DIR = Path("cad/base/Yjunction")

# ── Baseline RF surface outline (19 vertices, generator µm coords) ──────
# Extracted from rf_surface_y.step.
# Mapping: x_gen = x_step(µm) - 2500, y_gen = z_step(µm) + 2500
BASELINE_RF = [
    (-50.0, -80.0),    # [0]  o3: arm 0, -hw
    (-5.7, -3.3),      # [1]  neck: sector arm2↔arm0
    (-94.3, -3.3),     # [2]  o3: arm 2, +hw
    (-354.1, 146.7),   # [3]  arm 2 tip, +hw
    (-304.1, 233.3),   # [4]  arm 2 tip, -hw
    (-44.3, 83.3),     # [5]  o3: arm 2, -hw
    (-0.1, 6.7),       # [6]  neck: sector arm1↔arm2
    (44.2, 83.3),      # [7]  o3: arm 1, +hw
    (312.7, 238.3),    # [8]  arm 1 + extension
    (255.7, 337.0),    # [9]  extension
    (429.0, 437.0),    # [10] extension
    (593.0, 153.0),    # [11] extension
    (419.7, 53.0),     # [12] extension
    (362.7, 151.7),    # [13] extension
    (94.3, -3.3),      # [14] o3: arm 1, -hw
    (5.7, -3.3),       # [15] neck: sector arm0↔arm1
    (50.0, -80.0),     # [16] o3: arm 0, +hw
    (50.0, -380.0),    # [17] arm 0 tip, +hw
    (-50.0, -380.0),   # [18] arm 0 tip, -hw
]

# Sector definitions for the 3 taper regions.
# Each sector spans the junction between two adjacent arms.
# The polygon path traverses sectors in reverse of their natural
# direction (natural = arm_a +hw → arm_b -hw).
# Format: (o3_start_idx, o3_end_idx, sector_angle_a, sector_angle_b)
_SECTOR_DEFS = [
    (0, 2, 150.0, -90.0),    # bl[0]→bl[2]: sector arm2↔arm0
    (5, 7, 30.0, 150.0),     # bl[5]→bl[7]: sector arm1↔arm2
    (14, 16, -90.0, 30.0),   # bl[14]→bl[16]: sector arm0↔arm1
]

# Fixed arm segments between taper regions (baseline vertex indices)
_ARM_SEGS = [
    [3, 4, 5],               # arm 2 body
    list(range(8, 15)),       # arm 1 + extension
    [17, 18],                 # arm 0 body
]

# ── Defaults (from base STEP files) ─────────────────────────────────────
DEFAULTS = dict(
    electrode_gap_um=10.0,

    o1_x_um=3.3,
    o1_y_um=5.7,
    o1_r_um=0.0,

    o2_x_um=80.0,       # default = o3_x: taper goes directly from arm to neck
    o2_r_um=0.0,

    o3_x_um=80.0,
    o3_y_um=50.0,

    branch_rotation_deg=-90.0,
    arm_length_um=380.0,

    dc_inner_edge_um=60.0,
    dc_outer_edge_um=164.0,
    dc_first_center_um=127.5,
    dc_segment_length_um=55.0,
    dc_regular_length_um=70.0,
    dc_segment_gap_um=5.0,
    dc_segments_per_side=4,
    dc_min_width_um=20.0,
    dc_junction_pad=True,

    ext_arm_index=1,
    ext_arm_length_um=390.0,

    large_dc_arm_index=-1,
    large_dc_along_start_um=0.0,
    large_dc_along_end_um=0.0,
    large_dc_inner_um=0.0,
    large_dc_outer_um=0.0,
)


# ── Rotation / transform helpers ────────────────────────────────────────

def _rotate_pt(x, y, angle_deg):
    a = math.radians(angle_deg)
    return x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)


def _rotate_pts(pts, angle_deg):
    return [_rotate_pt(x, y, angle_deg) for x, y in pts]


# ── Arc rounding ────────────────────────────────────────────────────────

def _arc_round(prev, corner, nxt, radius, n_pts=16):
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
    dot = max(-1.0, min(1.0, u1x * u2x + u1y * u2y))
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
    cross = ((t1[0] - arc_cx) * (t2[1] - arc_cy) -
             (t1[1] - arc_cy) * (t2[0] - arc_cx))
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


# ── RF polygon builder ──────────────────────────────────────────────────

def _build_sector_taper(p, o3_start, o3_end, sector_angle_a, sector_angle_b):
    """Build taper curve from o3_start to o3_end.

    The polygon path traverses each sector in reverse compared to the
    sector's natural definition. In the reversed direction the path is:
      o3_start → o2_b → o1 → o2_a → o3_end

    sector_angle_a defines the coordinate frame for o1 and o2_a.
    sector_angle_b defines the coordinate frame for o2_b.
    """
    o3_y = p["o3_y_um"]
    o1 = _rotate_pt(p["o1_x_um"], p["o1_y_um"], sector_angle_a)
    o2_a = _rotate_pt(p["o2_x_um"], o3_y, sector_angle_a)
    o2_b = _rotate_pt(p["o2_x_um"], -o3_y, sector_angle_b)
    r1, r2 = p["o1_r_um"], p["o2_r_um"]

    if r1 > 0 or r2 > 0:
        pts = [o3_start]
        pts.extend(_arc_round(o3_start, o2_b, o1, r2))
        pts.extend(_arc_round(o2_b, o1, o2_a, r1))
        pts.extend(_arc_round(o1, o2_a, o3_end, r2))
        pts.append(o3_end)
        return pts

    return [o3_start, o2_b, o1, o2_a, o3_end]


def build_rf_polygon(p: dict, baseline=None) -> SPoly:
    """Build RF Y-junction polygon.

    Constructs the polygon by walking through the baseline outline,
    replacing each taper region with parameterized control-point curves.
    Fixed arm/extension vertices come directly from the baseline.
    """
    bl = list(baseline or BASELINE_RF)

    pts = []
    for i, ((s_idx, e_idx, sa, sb), arm_seg) in enumerate(
            zip(_SECTOR_DEFS, _ARM_SEGS)):
        taper = _build_sector_taper(p, bl[s_idx], bl[e_idx], sa, sb)
        if i == 0:
            pts.extend(taper)
        else:
            pts.extend(taper[1:])
        pts.extend(bl[j] for j in arm_seg)

    poly = SPoly(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


# ── DC electrode builder ────────────────────────────────────────────────

def _dc_rect_along_branch(along_start, along_end, cross_inner, cross_outer,
                           branch_angle_deg):
    rect = [
        (along_start, cross_inner),
        (along_end, cross_inner),
        (along_end, cross_outer),
        (along_start, cross_outer),
    ]
    return _rotate_pts(rect, branch_angle_deg)


def build_dc_polygons(p: dict, rf_poly: SPoly) -> list[list[tuple[float, float]]]:
    """Build DC electrode polygons along each branch, clipped against RF keepout."""
    gap = p["electrode_gap_um"]
    rot = p["branch_rotation_deg"]
    arm_len = p["arm_length_um"]
    dc_inner = p["dc_inner_edge_um"]
    dc_outer = p["dc_outer_edge_um"]
    first_center = p["dc_first_center_um"]
    first_len = p["dc_segment_length_um"]
    reg_len = p["dc_regular_length_um"]
    seg_gap = p["dc_segment_gap_um"]
    n_segs = int(p["dc_segments_per_side"])
    min_w = p["dc_min_width_um"]
    ext_idx = int(p.get("ext_arm_index", -1))
    ext_arm_len = p.get("ext_arm_length_um", arm_len)

    rf_keepout = rf_poly.buffer(gap, join_style="round")

    branch_angles = [rot + i * 120.0 for i in range(3)]
    candidates = []

    for bi, ba in enumerate(branch_angles):
        this_arm_len = ext_arm_len if bi == ext_idx else arm_len
        cursor = first_center - first_len / 2.0
        for seg_i in range(n_segs):
            seg_len = first_len if seg_i == 0 else reg_len
            a_start = cursor
            a_end = cursor + seg_len
            if a_end > this_arm_len + 0.5:
                break
            candidates.append(
                _dc_rect_along_branch(a_start, a_end, dc_inner, dc_outer, ba))
            candidates.append(
                _dc_rect_along_branch(a_start, a_end, -dc_outer, -dc_inner, ba))
            cursor = a_end + seg_gap

    if p.get("dc_junction_pad", True):
        along_lim = first_center - first_len / 2.0 - seg_gap
        for i in range(3):
            angle_a = branch_angles[i]
            angle_b = branch_angles[(i + 1) % 3]
            pad = [
                (0.0, 0.0),
                _rotate_pt(along_lim, dc_inner, angle_a),
                _rotate_pt(along_lim, dc_outer, angle_a),
                _rotate_pt(along_lim, -dc_outer, angle_b),
                _rotate_pt(along_lim, -dc_inner, angle_b),
            ]
            candidates.append(pad)

    ldc_idx = int(p.get("large_dc_arm_index", -1))
    if ldc_idx >= 0:
        ldc_start = p.get("large_dc_along_start_um", 0)
        ldc_end = p.get("large_dc_along_end_um", 0)
        ldc_inner = p.get("large_dc_inner_um", 0)
        ldc_outer = p.get("large_dc_outer_um", 0)
        if ldc_end > ldc_start:
            ba_ldc = branch_angles[ldc_idx]
            candidates.append(_dc_rect_along_branch(
                ldc_start, ldc_end, ldc_inner, ldc_outer, ba_ldc))

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
            shrunk = poly.buffer(-min_w / 2.0)
            if shrunk.is_empty:
                continue
            coords = list(poly.exterior.coords)
            if coords[-1] == coords[0]:
                coords = coords[:-1]
            if len(coords) >= 3:
                result.append(coords)

    return result


# ── Validation ──────────────────────────────────────────────────────────

def validate(p: dict, rf_poly: SPoly,
             dc_polys: list[list[tuple[float, float]]]) -> list[str]:
    errors = []
    gap = p["electrode_gap_um"]
    dc_min_w = p["dc_min_width_um"]

    if isinstance(rf_poly, MultiPolygon):
        errors.append(f"RF is MultiPolygon with {len(rf_poly.geoms)} parts")
    if not rf_poly.is_valid:
        from shapely.validation import explain_validity
        errors.append(f"RF polygon invalid: {explain_validity(rf_poly)}")

    for i, verts in enumerate(dc_polys):
        dc = SPoly(verts)
        d = rf_poly.distance(dc)
        if d < gap - 0.5:
            errors.append(f"RF-DC[{i}] gap = {d:.2f} < {gap} µm")

    dc_gap = p.get("dc_segment_gap_um", gap)
    dc_shapes = [SPoly(v) for v in dc_polys]
    for i in range(len(dc_shapes)):
        for j in range(i + 1, len(dc_shapes)):
            d = dc_shapes[i].distance(dc_shapes[j])
            if d < dc_gap - 0.5 and d > 0.01:
                errors.append(f"DC[{i}]-DC[{j}] gap = {d:.2f} < {dc_gap} µm")

    for i, verts in enumerate(dc_polys):
        dc = SPoly(verts)
        shrunk = dc.buffer(-dc_min_w / 2.0)
        if shrunk.is_empty:
            errors.append(f"DC[{i}] narrower than {dc_min_w} µm")

    return errors


# ── SVG export ──────────────────────────────────────────────────────────

def write_svg(rf_poly: SPoly,
              dc_polys: list[list[tuple[float, float]]],
              path: str, p: dict) -> None:
    margin = 40
    max_reach = max(
        p["arm_length_um"],
        p.get("ext_arm_length_um", 0),
        p.get("large_dc_along_end_um", 0),
        p.get("dc_outer_edge_um", 164.0),
    )
    # Check baseline extension vertices
    for vx, vy in BASELINE_RF:
        max_reach = max(max_reach, math.hypot(vx, vy))
    extent = max_reach + margin
    size = 2 * extent
    ox, oy = extent, extent

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{size:.0f}" height="{size:.0f}" viewBox="0 0 {size:.0f} {size:.0f}">',
        f'<rect width="{size:.0f}" height="{size:.0f}" fill="#f8f8f8"/>',
        f'<line x1="{ox:.0f}" y1="0" x2="{ox:.0f}" y2="{size:.0f}" '
        f'stroke="#ddd" stroke-width="0.5"/>',
        f'<line x1="0" y1="{oy:.0f}" x2="{size:.0f}" y2="{oy:.0f}" '
        f'stroke="#ddd" stroke-width="0.5"/>',
    ]

    def svg_poly(verts, fill, stroke="#333", sw=0.3, opacity=0.8):
        pts = " ".join(f"{ox + v[0]:.2f},{oy - v[1]:.2f}" for v in verts)
        return (f'<polygon points="{pts}" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')

    def svg_path_ring(exterior_coords, interior_rings, fill):
        def ring_to_d(coords):
            parts = []
            for i, (x, y) in enumerate(coords):
                cmd = "M" if i == 0 else "L"
                parts.append(f"{cmd}{ox + x:.2f},{oy - y:.2f}")
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

    for verts in dc_polys:
        lines.append(svg_poly(verts, "#6699ff"))

    sb_y = oy + extent - margin + 10
    lines.append(f'<line x1="{ox:.0f}" y1="{sb_y:.0f}" '
                 f'x2="{ox + 100:.0f}" y2="{sb_y:.0f}" '
                 f'stroke="black" stroke-width="1"/>')
    lines.append(f'<text x="{ox + 50:.0f}" y="{sb_y + 15:.0f}" '
                 f'text-anchor="middle" font-size="10" '
                 f'font-family="monospace">100 µm</text>')

    ly = 15
    for label, color in [("RF", "#ff6666"), ("DC", "#6699ff")]:
        lines.append(f'<rect x="5" y="{ly}" width="12" height="8" '
                     f'fill="{color}" stroke="#333" stroke-width="0.3"/>')
        lines.append(f'<text x="22" y="{ly + 8}" font-size="9" '
                     f'font-family="monospace">{label}</text>')
        ly += 14

    lines.append('</svg>')
    Path(path).write_text('\n'.join(lines))


# ── PNG export ──────────────────────────────────────────────────────────

def write_png(rf_poly: SPoly,
              dc_polys: list[list[tuple[float, float]]],
              path: str, p: dict) -> None:
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

    def add_polys(verts_list, color, label=None):
        patches = [MPoly(v, closed=True) for v in verts_list]
        if patches:
            pc = PatchCollection(patches, facecolor=color, edgecolor="#333",
                                 linewidth=0.3, alpha=0.8, label=label)
            ax.add_collection(pc)

    add_polys([list(rf_poly.exterior.coords)], "#ff6666", "RF")
    add_polys(dc_polys, "#6699ff", "DC")

    max_reach = max(p["arm_length_um"], p.get("dc_outer_edge_um", 164.0))
    for vx, vy in BASELINE_RF:
        max_reach = max(max_reach, math.hypot(vx, vy))
    lim = max_reach + 40
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title("Y-junction surface layout")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.grid(True, alpha=0.3)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── STEP export ─────────────────────────────────────────────────────────

def _clean_verts(verts, tol=0.1):
    cleaned = []
    n = len(verts)
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        if math.hypot(x1 - x0, y1 - y0) < tol:
            continue
        if cleaned:
            px, py = cleaned[-1]
            cross = (x0 - px) * (y1 - y0) - (y0 - py) * (x1 - x0)
            if abs(cross) < tol * tol:
                cleaned[-1] = (x0, y0)
                continue
        cleaned.append((x0, y0))
    return cleaned if len(cleaned) >= 3 else verts


def write_step(rf_poly: SPoly,
               dc_polys: list[list[tuple[float, float]]],
               out_dir: Path,
               p: dict | None = None) -> None:
    """Write STEP files in STEP native coordinates.

    RF output combines the modified surface electrode with structural
    geometry from rf_3d_y.step. All shapes share the same coordinate
    frame as the base STEP files.
    """
    try:
        from OCP.gp import gp_Pnt, gp_Vec
        from OCP.BRepBuilderAPI import (
            BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire,
            BRepBuilderAPI_MakeFace)
        from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
        from OCP.TopoDS import TopoDS_Compound, TopoDS
        from OCP.BRep import BRep_Builder
        from OCP.STEPControl import (STEPControl_Writer, STEPControl_AsIs,
                                      STEPControl_Reader)
        from OCP.Interface import Interface_Static
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_SOLID
    except ImportError:
        print("  OCP not available — skipping STEP export")
        return

    def to_step_mm(x_gen, y_gen):
        return (x_gen + 2500) * 1e-3, (y_gen - 2500) * 1e-3

    def extrude_step(verts_gen, y_bot_um=SURFACE_Y_BOT_UM):
        verts_gen = _clean_verts(verts_gen)
        n = len(verts_gen)
        y_bot = y_bot_um * 1e-3
        thickness_mm = THICKNESS_UM * 1e-3
        wb = BRepBuilderAPI_MakeWire()
        for i in range(n):
            x0, z0 = to_step_mm(*verts_gen[i])
            x1, z1 = to_step_mm(*verts_gen[(i + 1) % n])
            if math.hypot(x1 - x0, z1 - z0) < 1e-7:
                continue
            edge = BRepBuilderAPI_MakeEdge(
                gp_Pnt(x0, y_bot, z0),
                gp_Pnt(x1, y_bot, z1)).Edge()
            wb.Add(edge)
        face = BRepBuilderAPI_MakeFace(wb.Wire(), True).Face()
        return BRepPrimAPI_MakePrism(
            face, gp_Vec(0, thickness_mm, 0)).Shape()

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

    def load_step_solids(path):
        reader = STEPControl_Reader()
        reader.ReadFile(str(path))
        reader.TransferRoots()
        shape = reader.OneShape()
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        solids = []
        while exp.More():
            solids.append(TopoDS.Solid_s(exp.Current()))
            exp.Next()
        return solids

    # RF surface
    rf_coords = list(rf_poly.exterior.coords)
    if rf_coords[-1] == rf_coords[0]:
        rf_coords = rf_coords[:-1]
    rf_surface = extrude_step(rf_coords)

    # Load rf_3d_y.step
    rf_3d_path = BASE_DIR / "rf_3d_y.step"
    rf_3d_solids = []
    if rf_3d_path.exists():
        rf_3d_solids = load_step_solids(rf_3d_path)
        print(f"  Loaded {len(rf_3d_solids)} solids from {rf_3d_path}")

    rf_all = [rf_surface] + rf_3d_solids
    rf_compound = make_compound(rf_all)
    write(rf_compound, out_dir / "rf.step")
    print(f"  Wrote {out_dir / 'rf.step'}")

    # DC
    dc_shapes = []
    if dc_polys:
        dc_shapes = [extrude_step(v) for v in dc_polys]
        write(make_compound(dc_shapes), out_dir / "dc.step")
        print(f"  Wrote {out_dir / 'dc.step'}")

    # Combined
    all_shapes = rf_all + dc_shapes
    write(make_compound(all_shapes), out_dir / "combined.step")
    print(f"  Wrote {out_dir / 'combined.step'}")


# ── Main pipeline ───────────────────────────────────────────────────────

def generate(p: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Y-junction surface generator")
    print(f"{'='*60}")
    print(f"  Electrode gap: {p['electrode_gap_um']} µm")
    print(f"  O1: ({p['o1_x_um']}, {p['o1_y_um']})  r={p['o1_r_um']}")
    print(f"  O2: x={p['o2_x_um']}  r={p['o2_r_um']}  [y=o3_y={p['o3_y_um']}]")
    print(f"  O3: ({p['o3_x_um']}, {p['o3_y_um']})  [fixed from baseline]")

    rf_poly = build_rf_polygon(p)
    print(f"\n  RF Y-body: area={rf_poly.area:.1f} µm², "
          f"valid={rf_poly.is_valid}, type={rf_poly.geom_type}")

    dc_polys = build_dc_polygons(p, rf_poly)
    print(f"  DC segments: {len(dc_polys)}")

    errors = validate(p, rf_poly, dc_polys)
    if errors:
        print(f"\n  VALIDATION ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
    else:
        print(f"\n  All validation checks passed.")

    out_dir = Path(p.get("out_dir", "cad/generated/y_junction"))
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = p.get("out_svg", str(out_dir / "layout.svg"))
    write_svg(rf_poly, dc_polys, svg_path, p)
    print(f"  Wrote {svg_path}")

    png_path = p.get("out_png", str(out_dir / "layout.png"))
    write_png(rf_poly, dc_polys, png_path, p)

    write_step(rf_poly, dc_polys, out_dir, p=p)

    params_out = {k: v for k, v in p.items()
                  if k not in ("out_dir", "out_svg", "out_png", "validate")}
    params_path = str(out_dir / "params.json")
    Path(params_path).write_text(json.dumps(params_out, indent=2) + "\n")
    print(f"  Wrote {params_path}")

    if errors and p.get("validate", False):
        sys.exit(1)

    return {"rf_poly": rf_poly, "dc_polys": dc_polys, "errors": errors}


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parameterized Y-junction surface electrode generator")

    ap.add_argument("--electrode-gap-um", type=float,
                    default=DEFAULTS["electrode_gap_um"])
    ap.add_argument("--o1-x-um", type=float, default=DEFAULTS["o1_x_um"])
    ap.add_argument("--o1-y-um", type=float, default=DEFAULTS["o1_y_um"],
                    help="Neck half-width (µm) — primary variable")
    ap.add_argument("--o1-r-um", type=float, default=DEFAULTS["o1_r_um"],
                    help="Curvature radius at neck (µm) — primary variable")
    ap.add_argument("--o2-x-um", type=float, default=DEFAULTS["o2_x_um"],
                    help="Taper-start distance from center (µm) — primary variable")
    ap.add_argument("--o2-r-um", type=float, default=DEFAULTS["o2_r_um"],
                    help="Curvature at taper-start (µm) — primary variable")
    ap.add_argument("--dc-inner-edge-um", type=float,
                    default=DEFAULTS["dc_inner_edge_um"])
    ap.add_argument("--dc-outer-edge-um", type=float,
                    default=DEFAULTS["dc_outer_edge_um"])
    ap.add_argument("--dc-first-center-um", type=float,
                    default=DEFAULTS["dc_first_center_um"])
    ap.add_argument("--dc-segment-length-um", type=float,
                    default=DEFAULTS["dc_segment_length_um"])
    ap.add_argument("--dc-regular-length-um", type=float,
                    default=DEFAULTS["dc_regular_length_um"])
    ap.add_argument("--dc-segment-gap-um", type=float,
                    default=DEFAULTS["dc_segment_gap_um"])
    ap.add_argument("--dc-segments-per-side", type=int,
                    default=DEFAULTS["dc_segments_per_side"])
    ap.add_argument("--dc-min-width-um", type=float,
                    default=DEFAULTS["dc_min_width_um"])
    ap.add_argument("--no-junction-pad", action="store_true")
    ap.add_argument("--out-dir", type=str,
                    default="cad/generated/y_junction")
    ap.add_argument("--out-svg", type=str, default=None)
    ap.add_argument("--out-png", type=str, default=None)
    ap.add_argument("--validate", action="store_true")

    args = ap.parse_args()

    p = dict(DEFAULTS)
    for k in DEFAULTS:
        cli_name = k
        if hasattr(args, cli_name):
            p[k] = getattr(args, cli_name)
    p["dc_junction_pad"] = not args.no_junction_pad
    p["out_dir"] = args.out_dir
    p["validate"] = args.validate
    if args.out_svg:
        p["out_svg"] = args.out_svg
    if args.out_png:
        p["out_png"] = args.out_png

    generate(p)


if __name__ == "__main__":
    main()
