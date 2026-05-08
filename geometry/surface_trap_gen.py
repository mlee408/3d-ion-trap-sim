#!/usr/bin/env python3
"""
surface_trap_gen.py — Parametric STEP generator for a two-rail surface X-junction ion trap.

Generates RF, DC, and Ground electrode STEP files for a surface-electrode
X-junction trap.  All user-facing dimensions are in micrometres; OCC uses mm.

Uses raw OCP (OpenCascade Python bindings) directly — no CadQuery dependency.
Uses Shapely for 2D polygon operations (buffer/clip for clearance enforcement).

RF geometry (two-rail + junction caps)
--------------------------------------
  Top tier (z = -thickness .. 0):
    4 L-shaped quadrant rail pieces (each covering two adjacent arm rails)
    + 4 triangular junction cap pieces (tapering the inner RF edge toward center).

  Bottom tier (z = -2*thickness .. -thickness):
    Square frame from ±(cell_half - substrate_margin) to
    ±(cell_half + substrate_margin).

  Baseline values from cad/surface/rf.step:
    rf_rail_width=60, center_gap=100, o1_rf=30
    outer RF edge at 110, inner at 50 (arms), taper to 30 at junction
    substrate_margin=28  ->  frame ±272 to ±328

DC geometry
-----------
  Outer DC (flanking outside the RF rails):
    • 4 corner pads (one per quadrant, dc_inner × dc_inner squares)
    • Arm-flank pads (n_dc_flank per arm per side)
    dc_inner = center_gap/2 + rf_rail_width + gap

  Center DC (segmented control electrodes in center gap):
    • 1 junction center pad (square, bounded by RF caps)
    • Segmented arm center strips (one series per arm)
    center_strip_width = center_gap - 2*gap

Ground geometry
---------------
  • 4 L-shaped patches (top tier, outside DC flanks)
  • 1 solid substrate base (bottom tier ±(cell_half - substrate_margin))

Sweep parameters (exposed via CLI):
  rf_rail_width, center_gap, o1_rf

  o1_rf = RF inner boundary at junction (diagonal vertex distance from centre)
          Controls where the triangular junction cap vertex is placed.
          Must be < center_gap/2.

Fixed constraints (internal, not on CLI):
  gap=10, cell_half=300, thickness=10, substrate_margin=28,
  dc_flank_width=60, dc_flank_len=40, dc_seg_gap=10, n_dc_flank=2,
  center_seg_len=40, junction_center_half=20

Usage
-----
    python geometry/surface_trap_gen.py
    python geometry/surface_trap_gen.py --rf-rail-width 60 --center-gap 100 --o1-rf 30
    python geometry/surface_trap_gen.py --out-dir cad/generated/surface
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from OCP.gp import gp_Pnt, gp_Dir, gp_Vec, gp_Ax2
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
)
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeBox
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.TopoDS import TopoDS_Shape, TopoDS_Compound
from OCP.BRep import BRep_Builder
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.Interface import Interface_Static
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
from shapely.ops import unary_union

# ─────────────────────────────────────────────────────────────────────────────
# Default parameters (µm unless noted)
# ─────────────────────────────────────────────────────────────────────────────
SWEEP_PARAMS: dict = dict(
    rf_rail_width      = 60.0,    # width of one RF rail [µm]
    center_gap         = 100.0,   # gap between inner RF edges on straight arms [µm]
    o1_rf              = 30.0,    # RF inner vertex at junction (diagonal dist) [µm]
)

FIXED_PARAMS: dict = dict(
    cell_half          = 300.0,   # half cell size [µm]
    gap                = 10.0,    # electrode–electrode clearance [µm]
    thickness          = 10.0,    # electrode layer thickness [µm]
    substrate_margin   = 28.0,    # frame half-width around cell_half [µm]
    dc_flank_width     = 60.0,    # transverse width of outer DC flank pads [µm]
    dc_flank_len       = 40.0,    # axial length of outer DC flank pads [µm]
    dc_seg_gap         = 10.0,    # gap between adjacent DC segments [µm]
    n_dc_flank         = 2,       # flank pads per arm per side
    center_seg_len     = 40.0,    # axial length of center DC segments [µm]
    junction_center_half = 20.0,  # half-size of junction center pad [µm]
)

DEFAULTS: dict = {**FIXED_PARAMS, **SWEEP_PARAMS}


def _mm(x: float) -> float:
    """µm → mm (OCC native unit)."""
    return x * 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Low-level OCC helpers
# ─────────────────────────────────────────────────────────────────────────────
def _make_box(x0: float, y0: float, x1: float, y1: float,
              z_bot: float, z_top: float) -> TopoDS_Shape:
    """Axis-aligned box from (x0,y0,z_bot) to (x1,y1,z_top), all in µm."""
    xlo, xhi = min(x0, x1), max(x0, x1)
    ylo, yhi = min(y0, y1), max(y0, y1)
    zlo, zhi = min(z_bot, z_top), max(z_bot, z_top)
    return BRepPrimAPI_MakeBox(
        gp_Pnt(_mm(xlo), _mm(ylo), _mm(zlo)),
        gp_Pnt(_mm(xhi), _mm(yhi), _mm(zhi)),
    ).Shape()


def _extrude_polygon(verts_um: list[tuple[float, float]],
                     z_bot: float, z_top: float) -> TopoDS_Shape:
    """Extrude a closed polygon (vertices in µm, CCW) from z_bot to z_top [µm]."""
    n = len(verts_um)
    wire_builder = BRepBuilderAPI_MakeWire()
    for i in range(n):
        x0, y0 = verts_um[i]
        x1, y1 = verts_um[(i + 1) % n]
        edge = BRepBuilderAPI_MakeEdge(
            gp_Pnt(_mm(x0), _mm(y0), _mm(z_bot)),
            gp_Pnt(_mm(x1), _mm(y1), _mm(z_bot)),
        ).Edge()
        wire_builder.Add(edge)
    wire = wire_builder.Wire()
    face = BRepBuilderAPI_MakeFace(wire, True).Face()
    h = _mm(z_top - z_bot)
    prism = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, h)).Shape()
    return prism


def _cut(a: TopoDS_Shape, b: TopoDS_Shape) -> TopoDS_Shape:
    return BRepAlgoAPI_Cut(a, b).Shape()


def _write_step(shape: TopoDS_Shape, path: str) -> None:
    writer = STEPControl_Writer()
    Interface_Static.SetCVal_s("write.step.schema", "AP203")
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(path)
    if status != 1:
        raise RuntimeError(f"STEP write failed with status {status}: {path}")


def _make_compound(shapes: list[TopoDS_Shape]) -> TopoDS_Shape:
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for s in shapes:
        builder.Add(compound, s)
    return compound


def _shape_bbox(shape: TopoDS_Shape) -> tuple[float, float, float, float, float, float]:
    """Return (xmin, ymin, zmin, xmax, ymax, zmax) in µm."""
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    x0, y0, z0, x1, y1, z1 = box.Get()
    return x0 * 1000, y0 * 1000, z0 * 1000, x1 * 1000, y1 * 1000, z1 * 1000


# ─────────────────────────────────────────────────────────────────────────────
# Derived quantities
# ─────────────────────────────────────────────────────────────────────────────
def _derived(p: dict) -> dict:
    """Compute commonly-used derived dimensions."""
    gi = p['center_gap'] / 2        # inner RF edge on arms (from center)
    go = gi + p['rf_rail_width']    # outer RF edge on arms (from center)
    o1 = p['o1_rf']                 # junction inner vertex (diagonal)
    gap = p['gap']
    C = p['cell_half']
    dc_inner = go + gap             # inner edge of outer DC flanks
    dc_outer = dc_inner + p['dc_flank_width']
    center_w_half = gi - gap        # half-width of center strip (from axis)
    return dict(
        gi=gi, go=go, o1=o1, gap=gap, C=C,
        dc_inner=dc_inner, dc_outer=dc_outer,
        center_w_half=center_w_half,
    )


def _validate(p: dict) -> None:
    d = _derived(p)
    gi, go, o1, C, gap = d['gi'], d['go'], d['o1'], d['C'], d['gap']

    if o1 >= gi:
        raise ValueError(
            f"o1_rf ({o1}) must be < center_gap/2 ({gi})")
    if o1 <= 0:
        raise ValueError(f"o1_rf ({o1}) must be > 0")
    if go >= C:
        raise ValueError(
            f"outer RF edge ({go}) must be < cell_half ({C})")
    if p['rf_rail_width'] <= 0:
        raise ValueError("rf_rail_width must be > 0")
    if p['center_gap'] <= 2 * gap:
        raise ValueError(
            f"center_gap ({p['center_gap']}) must be > 2*gap ({2*gap})")
    if d['center_w_half'] <= 0:
        raise ValueError(
            f"center strip half-width ({d['center_w_half']}) must be > 0 "
            f"(center_gap/2 - gap = {gi} - {gap})")


# ─────────────────────────────────────────────────────────────────────────────
# 2D polygon definitions (all in µm, z=0 plane)
# ─────────────────────────────────────────────────────────────────────────────
def _rf_polygons_2d(p: dict) -> list[list[tuple[float, float]]]:
    """
    Return the 2D polygon outlines for all RF pieces (top tier).

    8 pieces total:
      4 L-shaped quadrant rails  (6 vertices each)
      4 triangular junction caps (3 vertices each)
    """
    d = _derived(p)
    gi, go, o1, C = d['gi'], d['go'], d['o1'], d['C']

    polys = []
    for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
        # L-shape: covers two arm rails meeting at the quadrant corner
        #   For +x,+y: (+x arm upper rail) + (+y arm right rail) + shared corner
        l_verts = [
            (sx * gi,  sy * gi),
            (sx * C,   sy * gi),
            (sx * C,   sy * go),
            (sx * go,  sy * go),
            (sx * go,  sy * C),
            (sx * gi,  sy * C),
        ]
        # Ensure CCW winding
        if _signed_area(l_verts) < 0:
            l_verts = list(reversed(l_verts))
        polys.append(l_verts)

        # Triangular junction cap: extends inner RF edge from gi toward o1
        t_verts = [
            (sx * o1, sy * o1),
            (sx * o1, sy * gi),
            (sx * gi, sy * o1),
        ]
        if _signed_area(t_verts) < 0:
            t_verts = list(reversed(t_verts))
        polys.append(t_verts)

    return polys


def _signed_area(verts: list[tuple[float, float]]) -> float:
    """Signed area of a polygon (positive = CCW)."""
    n = len(verts)
    s = 0.0
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return s / 2.0


def _rf_union_shapely(p: dict) -> ShapelyPolygon:
    """Build the Shapely union of all RF polygons (for buffering / distance checks)."""
    polys = _rf_polygons_2d(p)
    shapely_polys = [ShapelyPolygon(v) for v in polys]
    return unary_union(shapely_polys)


def _rf_exclusion_shapely(p: dict) -> ShapelyPolygon:
    """RF union buffered outward by gap — the exclusion zone for DC pads."""
    rf = _rf_union_shapely(p)
    return rf.buffer(p['gap'], join_style='mitre', mitre_limit=10.0)


# ─────────────────────────────────────────────────────────────────────────────
# Outer DC pad polygons (2D)
# ─────────────────────────────────────────────────────────────────────────────
def _outer_dc_polygons_2d(p: dict) -> list[list[tuple[float, float]]]:
    """
    Outer DC pads: corner pads + arm flanks.

    Layout per quadrant (+x,+y example):
      Corner pad:  [dc_inner, dc_outer] × [dc_inner, dc_outer]
      +x arm flanks: x from dc_outer+seg_gap outward, y in [dc_inner, dc_outer]
      +y arm flanks: y from dc_outer+seg_gap outward, x in [dc_inner, dc_outer]
    """
    d = _derived(p)
    di, do = d['dc_inner'], d['dc_outer']
    sg = p['dc_seg_gap']
    fl = p['dc_flank_len']
    n = p['n_dc_flank']
    C = d['C']
    gap = d['gap']
    cell_clip = C - gap / 2

    polys = []

    for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
        # Corner pad
        cx0, cy0 = sx * di, sy * di
        cx1, cy1 = sx * do, sy * do
        verts = _rect_verts(cx0, cy0, cx1, cy1)
        polys.append(verts)

        # Arm flanks along the arm that runs in the sx direction
        # (e.g., +x arm for sx=1: pads at increasing |x|, fixed y range)
        flank_start = do + sg
        for i in range(n):
            ax_lo = flank_start + i * (fl + sg)
            ax_hi = ax_lo + fl
            if ax_lo >= cell_clip:
                break
            ax_hi = min(ax_hi, cell_clip)
            # Transverse range: [di, do] in the sy direction
            verts = _rect_verts(sx * ax_lo, sy * di, sx * ax_hi, sy * do)
            polys.append(verts)

        # Arm flanks along the arm that runs in the sy direction
        for i in range(n):
            ax_lo = do + sg + i * (fl + sg)
            ax_hi = ax_lo + fl
            if ax_lo >= cell_clip:
                break
            ax_hi = min(ax_hi, cell_clip)
            verts = _rect_verts(sx * di, sy * ax_lo, sx * do, sy * ax_hi)
            polys.append(verts)

    return polys


def _rect_verts(x0: float, y0: float, x1: float, y1: float
                ) -> list[tuple[float, float]]:
    """Return CCW rectangle vertices."""
    xlo, xhi = min(x0, x1), max(x0, x1)
    ylo, yhi = min(y0, y1), max(y0, y1)
    return [(xlo, ylo), (xhi, ylo), (xhi, yhi), (xlo, yhi)]


# ─────────────────────────────────────────────────────────────────────────────
# Center DC pad polygons (2D)
# ─────────────────────────────────────────────────────────────────────────────
def _center_dc_polygons_2d(p: dict) -> list[list[tuple[float, float]]]:
    """
    Center DC control electrodes in the gap between RF rails.

    - 1 junction center pad (square)
    - 4 arm center segment series (one per arm)
    """
    d = _derived(p)
    gi, gap, C = d['gi'], d['gap'], d['C']
    cw = d['center_w_half']          # half-width of center strip
    jch = p['junction_center_half']  # half-size of junction center pad
    sl = p['center_seg_len']
    sg = p['dc_seg_gap']
    cell_clip = C - gap / 2

    polys = []

    # Junction center pad
    polys.append(_rect_verts(-jch, -jch, jch, jch))

    # Arm center segments — one series per arm
    # Each arm's center strip runs from gi+gap to cell_clip along the arm axis
    arm_start = gi + gap  # axial start of first segment (past the junction cap zone)

    for axis, sign in [('y', 1), ('y', -1), ('x', 1), ('x', -1)]:
        pos = arm_start
        while pos + sl <= cell_clip + 0.01:
            seg_end = min(pos + sl, cell_clip)
            if axis == 'y':
                verts = _rect_verts(-cw, sign * pos, cw, sign * seg_end)
            else:
                verts = _rect_verts(sign * pos, -cw, sign * seg_end, cw)
            polys.append(verts)
            pos = seg_end + sg
            if seg_end >= cell_clip - 0.01:
                break

    return polys


# ─────────────────────────────────────────────────────────────────────────────
# Clearance enforcement: clip DC polygons against RF exclusion zone
# ─────────────────────────────────────────────────────────────────────────────
def _clip_dc_against_rf(dc_polys: list[list[tuple[float, float]]],
                        rf_exclusion: ShapelyPolygon,
                        ) -> list[list[tuple[float, float]]]:
    """
    Subtract the RF exclusion zone (RF + gap buffer) from each DC polygon.
    Returns the surviving polygon outlines.
    """
    clipped = []
    for verts in dc_polys:
        dc_shape = ShapelyPolygon(verts)
        result = dc_shape.difference(rf_exclusion)
        if result.is_empty:
            continue
        if isinstance(result, ShapelyPolygon):
            _collect_poly(result, clipped)
        elif isinstance(result, MultiPolygon):
            for poly in result.geoms:
                _collect_poly(poly, clipped)
    return clipped


def _collect_poly(poly: ShapelyPolygon,
                  out: list[list[tuple[float, float]]]) -> None:
    """Extract exterior coords from a Shapely polygon (drop closing duplicate)."""
    coords = list(poly.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) >= 3:
        out.append(coords)


# ─────────────────────────────────────────────────────────────────────────────
# Build 3D solids
# ─────────────────────────────────────────────────────────────────────────────
def build_rf(p: dict) -> tuple[list[TopoDS_Shape], list[TopoDS_Shape]]:
    """
    Build RF electrode solids.

    Returns (top_tier_pieces, [substrate_frame]).
    top_tier_pieces: 8 solids (4 L-shapes + 4 triangle caps).
    """
    _validate(p)
    t = p['thickness']
    z_bot, z_top = -t, 0.0

    rf_polys = _rf_polygons_2d(p)
    top_pieces = [_extrude_polygon(v, z_bot, z_top) for v in rf_polys]

    # Substrate frame (bottom tier)
    sm = p['substrate_margin']
    C = p['cell_half']
    inner = C - sm
    outer = C + sm
    frame = _cut(
        _make_box(-outer, -outer, outer, outer, -2 * t, -t),
        _make_box(-inner, -inner, inner, inner, -2 * t, -t),
    )
    return top_pieces, [frame]


def build_dc(p: dict) -> list[TopoDS_Shape]:
    """
    Build DC electrode pads with clearance enforcement.

    All DC polygons are clipped against the RF exclusion zone (RF + gap buffer)
    to guarantee the minimum clearance.
    """
    _validate(p)
    t = p['thickness']
    z_bot, z_top = -t, 0.0

    rf_exclusion = _rf_exclusion_shapely(p)

    outer_polys = _outer_dc_polygons_2d(p)
    center_polys = _center_dc_polygons_2d(p)
    all_dc = outer_polys + center_polys

    clipped = _clip_dc_against_rf(all_dc, rf_exclusion)

    return [_extrude_polygon(v, z_bot, z_top) for v in clipped]


def build_ground(p: dict) -> list[TopoDS_Shape]:
    """
    Build ground electrode solids.

    4 L-shaped patches (top tier) in the corners outside the DC flanks,
    + 1 substrate base (bottom tier).
    """
    _validate(p)
    d = _derived(p)
    C = d['C']
    t = p['thickness']
    gap = d['gap']
    do = d['dc_outer']
    sm = p['substrate_margin']

    gnd_inner = do + gap
    gnd_outer = C

    z_bot, z_top = -t, 0.0
    solids: list[TopoDS_Shape] = []

    for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
        verts = [
            (sx * gnd_inner, sy * gnd_inner),
            (sx * gnd_inner, sy * gnd_outer),
            (sx * gnd_outer, sy * gnd_outer),
            (sx * gnd_outer, sy * gnd_inner),
        ]
        if _signed_area(verts) < 0:
            verts = list(reversed(verts))
        solids.append(_extrude_polygon(verts, z_bot, z_top))

    # Substrate base (bottom tier)
    sub_half = C - sm
    solids.append(_make_box(-sub_half, -sub_half, sub_half, sub_half,
                            -2 * t, -t))
    return solids


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def _compute_min_rf_dc_distance(p: dict, dc_polys_clipped: list[list[tuple[float, float]]]) -> float:
    """Compute the minimum distance between RF polygons and clipped DC polygons (2D)."""
    rf_union = _rf_union_shapely(p)
    min_dist = float('inf')
    for verts in dc_polys_clipped:
        dc_shape = ShapelyPolygon(verts)
        dist = rf_union.distance(dc_shape)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def _print_diagnostics(p: dict, rf_pieces: list[TopoDS_Shape],
                       dc_pieces: list[TopoDS_Shape],
                       dc_polys_clipped: list[list[tuple[float, float]]]) -> None:
    """Print diagnostic info: piece counts, bboxes, min RF-DC distance."""
    d = _derived(p)
    print(f"\n[diagnostics]")
    print(f"  RF pieces:  {len(rf_pieces)} (4 L-shapes + 4 junction caps)")
    print(f"  DC pieces:  {len(dc_pieces)}")

    # RF bbox
    if rf_pieces:
        all_rf = _make_compound(rf_pieces)
        x0, y0, z0, x1, y1, z1 = _shape_bbox(all_rf)
        print(f"  RF bbox:    x=[{x0:.1f}, {x1:.1f}] y=[{y0:.1f}, {y1:.1f}] µm")

    # DC bbox
    if dc_pieces:
        all_dc = _make_compound(dc_pieces)
        x0, y0, z0, x1, y1, z1 = _shape_bbox(all_dc)
        print(f"  DC bbox:    x=[{x0:.1f}, {x1:.1f}] y=[{y0:.1f}, {y1:.1f}] µm")

    # Min RF-DC distance
    if dc_polys_clipped:
        min_dist = _compute_min_rf_dc_distance(p, dc_polys_clipped)
        status = "OK" if min_dist >= p['gap'] - 0.1 else "WARNING: below gap!"
        print(f"  Min RF–DC:  {min_dist:.2f} µm  ({status})")

    print(f"  Derived:    gi={d['gi']:.1f} go={d['go']:.1f} dc_inner={d['dc_inner']:.1f} "
          f"dc_outer={d['dc_outer']:.1f} center_w={2*d['center_w_half']:.1f} µm")


# ─────────────────────────────────────────────────────────────────────────────
# SVG debug output
# ─────────────────────────────────────────────────────────────────────────────
def _write_debug_svg(p: dict, dc_polys_clipped: list[list[tuple[float, float]]],
                     path: str) -> None:
    """Write a 2D top-view SVG of all electrodes for visual inspection."""
    d = _derived(p)
    C = d['C']
    margin = 40
    size = 2 * (C + margin)
    # SVG coordinate system: origin at center, y flipped
    ox, oy = C + margin, C + margin
    scale = 1.0

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">',
        f'<rect width="{size}" height="{size}" fill="#f8f8f8"/>',
        # Grid lines
        f'<line x1="{ox}" y1="0" x2="{ox}" y2="{size}" stroke="#ddd" stroke-width="0.5"/>',
        f'<line x1="0" y1="{oy}" x2="{size}" y2="{oy}" stroke="#ddd" stroke-width="0.5"/>',
    ]

    def _svg_poly(verts, fill, stroke="#333", sw=0.5, opacity=0.8):
        pts = " ".join(f"{ox + v[0]*scale:.2f},{oy - v[1]*scale:.2f}" for v in verts)
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'

    # RF polygons
    for verts in _rf_polygons_2d(p):
        lines.append(_svg_poly(verts, "#ff6666", sw=0.3))

    # RF exclusion zone (for debugging)
    rf_excl = _rf_exclusion_shapely(p)
    if hasattr(rf_excl, 'exterior'):
        excl_coords = list(rf_excl.exterior.coords)
        lines.append(_svg_poly(excl_coords, "none", stroke="#ff0000", sw=0.3, opacity=0.3))

    # DC polygons (clipped)
    for verts in dc_polys_clipped:
        lines.append(_svg_poly(verts, "#6699ff", sw=0.3))

    # Ground polygons
    gnd_inner = d['dc_outer'] + d['gap']
    gnd_outer = C
    for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
        gnd_verts = [
            (sx * gnd_inner, sy * gnd_inner),
            (sx * gnd_inner, sy * gnd_outer),
            (sx * gnd_outer, sy * gnd_outer),
            (sx * gnd_outer, sy * gnd_inner),
        ]
        lines.append(_svg_poly(gnd_verts, "#88cc88", sw=0.3))

    # Scale bar
    lines.append(f'<line x1="{ox+10}" y1="{oy+C+20}" x2="{ox+110}" y2="{oy+C+20}" '
                 f'stroke="black" stroke-width="1"/>')
    lines.append(f'<text x="{ox+60}" y="{oy+C+35}" text-anchor="middle" '
                 f'font-size="10" font-family="monospace">100 µm</text>')

    # Legend
    ly = 15
    for label, color in [("RF", "#ff6666"), ("DC", "#6699ff"), ("Ground", "#88cc88")]:
        lines.append(f'<rect x="5" y="{ly}" width="12" height="8" fill="{color}" stroke="#333" stroke-width="0.3"/>')
        lines.append(f'<text x="22" y="{ly+8}" font-size="9" font-family="monospace">{label}</text>')
        ly += 14

    lines.append('</svg>')
    Path(path).write_text('\n'.join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Build and export
# ─────────────────────────────────────────────────────────────────────────────
def build_and_export(p: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    d = _derived(p)
    print(
        f"[surface_trap_gen] Parameters (two-rail):\n"
        f"  cell={2*p['cell_half']:.0f}×{2*p['cell_half']:.0f} µm  "
        f"gap={p['gap']:.0f} µm  thickness={p['thickness']:.0f} µm\n"
        f"  rf_rail_width={p['rf_rail_width']:.1f} µm  "
        f"center_gap={p['center_gap']:.1f} µm  o1_rf={p['o1_rf']:.1f} µm\n"
        f"  inner RF edge (arms): ±{d['gi']:.1f} µm  "
        f"outer RF edge: ±{d['go']:.1f} µm\n"
        f"  DC inner={d['dc_inner']:.1f} µm  DC outer={d['dc_outer']:.1f} µm  "
        f"n_flank={p['n_dc_flank']}\n"
        f"  Center strip width={2*d['center_w_half']:.1f} µm  "
        f"junction_center={2*p['junction_center_half']:.0f} µm\n"
        f"  substrate=[{p['cell_half']-p['substrate_margin']:.0f}, "
        f"{p['cell_half']+p['substrate_margin']:.0f}] µm",
        flush=True,
    )

    # Build RF
    print("[surface_trap_gen] Building RF ...", flush=True)
    rf_top, rf_frame = build_rf(p)
    rf_all = rf_top + rf_frame
    rf_compound = _make_compound(rf_all)
    rf_path = str(out_dir / "rf.step")
    _write_step(rf_compound, rf_path)
    print(f"  {len(rf_top)} top-tier pieces + {len(rf_frame)} frame → {rf_path}")

    # Build DC (with clipping)
    print("[surface_trap_gen] Building DC pads ...", flush=True)
    rf_exclusion = _rf_exclusion_shapely(p)
    outer_polys = _outer_dc_polygons_2d(p)
    center_polys = _center_dc_polygons_2d(p)
    all_dc_polys = outer_polys + center_polys
    dc_polys_clipped = _clip_dc_against_rf(all_dc_polys, rf_exclusion)

    t = p['thickness']
    dc_pieces = [_extrude_polygon(v, -t, 0.0) for v in dc_polys_clipped]
    print(f"  {len(outer_polys)} outer + {len(center_polys)} center → "
          f"{len(dc_polys_clipped)} after clipping", flush=True)
    dc_compound = _make_compound(dc_pieces)
    dc_path = str(out_dir / "dc.step")
    _write_step(dc_compound, dc_path)
    print(f"  → {dc_path}")

    # Build ground
    print("[surface_trap_gen] Building ground ...", flush=True)
    gnd_parts = build_ground(p)
    print(f"  {len(gnd_parts)} solids", flush=True)
    gnd_compound = _make_compound(gnd_parts)
    gnd_path = str(out_dir / "ground.step")
    _write_step(gnd_compound, gnd_path)
    print(f"  → {gnd_path}")

    # Combined STEP
    print("[surface_trap_gen] Writing combined STEP ...", flush=True)
    combined = _make_compound(rf_all + dc_pieces + gnd_parts)
    combined_path = str(out_dir / "combined.step")
    _write_step(combined, combined_path)
    print(f"  → {combined_path}")

    # Debug SVG
    svg_path = str(out_dir / "debug_layout.svg")
    _write_debug_svg(p, dc_polys_clipped, svg_path)
    print(f"  → {svg_path}")

    # Diagnostics
    _print_diagnostics(p, rf_top, dc_pieces, dc_polys_clipped)

    print(f"\n[surface_trap_gen] Done.  Files in {out_dir}/", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Parametric STEP generator for two-rail surface X-junction ion trap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for key, default in SWEEP_PARAMS.items():
        flag = f"--{key.replace('_', '-')}"
        t = type(default)
        ap.add_argument(flag, type=t, default=default,
                        metavar=str(default),
                        help=f"[µm]")
    ap.add_argument(
        "--out-dir", default="cad/generated/surface",
        help="Output directory for STEP + SVG files",
    )
    return ap


def main(argv=None) -> int:
    args = _make_parser().parse_args(argv)
    p = dict(FIXED_PARAMS)
    p.update({k: getattr(args, k) for k in SWEEP_PARAMS})
    try:
        build_and_export(p, Path(args.out_dir))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
